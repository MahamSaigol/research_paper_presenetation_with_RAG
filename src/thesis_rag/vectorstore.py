"""
Wraps Chroma as the vector store: a local, persistent, embedded database
(not a server you run separately) — same "no extra infrastructure"
pattern as the rest of this pipeline.

Design choice: chunk IDs are deterministic (f"{paper_name}::{chunk_index}"),
not random UUIDs.
    A deterministic ID means re-embedding the same paper's same chunk
    overwrites it in place (Chroma's upsert) rather than creating a
    duplicate. Combined with delete_paper() before re-embedding a changed
    paper (see scripts/embed.py), this keeps the store from accumulating
    stale duplicate chunks when a paper's chunking changes between runs
    (e.g. after a Phase 2 code fix that changes chunk boundaries).

Design choice: metadata carries paper_name/section/page_start/page_end,
not the full Chunk object.
    Chroma's `where` filter only works against exactly the fields you
    store in metadata — this is what makes Phase 5's bibliography
    filtering (year, publisher, etc. — todo) and Phase 6's "give me all
    methodology chunks for this paper" queries possible without loading
    every chunk into memory to filter in Python.

Design choice: boilerplate/administrative text is filtered out of every
query's results, not just excluded by section label.
    Confirmed against a real paper: short, low-information text like
    "authors declare no competing interests," "journal homepage:
    www.elsevier.com," and "authors do not have permission to share
    data" was showing up as a deceptively *close* semantic match for
    substantive queries like "what method does this paper propose" —
    close enough to crowd out real content from the top-k. This is a
    known embedding-space effect: very short, generic text has little
    distinguishing signal, so it can end up moderately close to almost
    any query rather than clearly far from unrelated ones. Section-label
    filtering can't catch this, since this text lands in "other" mixed
    in with genuinely useful misclassified content — the boilerplate
    patterns below are pattern-matched directly against chunk text, and
    only applied to short chunks, so a substantive passage that happens
    to mention a DOI in passing isn't affected. Filtered here in
    vectorstore.py (not in each caller) so every caller — Phase 4's
    answer.py, Phase 6's slidegen.py, search_test.py — benefits without
    having to remember to do it themselves.

    This turned out to be a recurring category, not a one-off: a second
    real paper surfaced entirely different boilerplate (CRediT
    authorship contribution statements, an ethics/human-research
    declaration) that the original pattern list didn't cover, since
    exact back-matter wording varies by publisher. Patterns are added as
    real papers surface them; a more systemic fix (detecting standard
    back-matter section *headings* at extraction time, which are far
    more consistent across publishers than the body text that follows
    them, and excluding that whole section from embedding) would close
    this category more completely, at the cost of reprocessing the
    whole corpus — worth doing if this keeps recurring.
"""

from __future__ import annotations

import re
from typing import Any

import chromadb

from .chunk import Chunk
from .config import settings
from .embed import embed_passages, embed_query

_client = None

# Patterns seen in real papers' front-matter/back-matter boilerplate:
# competing-interest declarations, data-availability statements, journal
# masthead lines, copyright notices, and access-log stamps some PDF
# sources embed ("Authorized licensed use limited to..."). Kept to
# short-ish chunks only (_BOILERPLATE_MAX_CHARS) so this can't
# accidentally exclude a substantive passage that happens to mention a
# DOI in passing — but "short-ish" has to mean "short relative to this
# pipeline's actual chunk sizes," not an arbitrary small number. An
# earlier version set this to 250 chars based on hand-written test
# fixtures, not real chunks; tested against the real pipeline, that was
# far too low — Phase 2's chunks run up to ~400 tokens
# (chunk_max_tokens in config.py), comfortably 1500-2000+ characters, so
# a combined back-matter chunk (competing-interest + data-availability
# statements chunked together) never got short enough to even reach the
# pattern check. 2500 comfortably covers real chunk sizes with margin.
_BOILERPLATE_PATTERNS = [
    # \s+ instead of literal spaces — confirmed necessary against a real
    # paper, where "competing financial interests" had a line break
    # between "financial" and "interests" (PDF text extraction preserves
    # original line wrapping, and chunk text joins lines with "\n"). A
    # literal space in the pattern doesn't match across that newline; \s
    # matches any whitespace including it.
    re.compile(p, re.IGNORECASE)
    for p in [
        r"competing\s+(financial\s+)?interest",
        r"data\s+availability",
        r"permission\s+to\s+share\s+data",
        r"journal\s+homepage",
        r"contents\s+lists\s+available\s+at\s+sciencedirect",
        r"published\s+by\s+elsevier",
        r"all\s+rights\s+reserved",
        r"prepared\s+using\s+\w+\.cls",
        r"authorized\s+licensed\s+use\s+limited\s+to",
        r"downloaded\s+on\s+\w+\s+\d",
        # CRediT authorship contribution statements: a per-author list of
        # standardized role keywords ("Writing – original draft",
        # "Conceptualization", "Funding acquisition", etc.). A few of
        # these terms are distinctive enough alone to catch the pattern
        # without needing to match the whole author-role list verbatim.
        r"writing\s*[-–—]\s*original\s+draft",
        r"writing\s*[-–—]\s*review\s*(&|and)\s*editing",
        r"funding\s+acquisition",
        r"project\s+administration",
        r"formal\s+analysis",
        r"data\s+curation",
        # Ethics/human-or-animal-research declarations
        r"human\s+or\s+animal\s+research",
        r"institutional\s+review\s+board",
        r"ethics\s+(approval|statement|committee)",
        r"informed\s+consent",
    ]
]
_BOILERPLATE_MAX_CHARS = 2500

# How much extra to over-fetch from Chroma to compensate for boilerplate
# getting filtered out — without this, filtering could leave fewer than
# n_results genuine results even when better ones exist further down.
_OVERFETCH_MULTIPLIER = 4


def _is_boilerplate(text: str) -> bool:
    if len(text) > _BOILERPLATE_MAX_CHARS:
        return False
    return any(p.search(text) for p in _BOILERPLATE_PATTERNS)


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client.get_or_create_collection(settings.collection_name)


def upsert_chunks(chunks: list[Chunk]) -> None:
    """Embeds and stores/overwrites the given chunks."""
    if not chunks:
        return
    collection = _get_collection()
    embeddings = embed_passages([c.text for c in chunks])
    ids = [f"{c.paper_name}::{c.chunk_index}" for c in chunks]
    metadatas: list[dict[str, Any]] = [
        {
            "paper_name": c.paper_name,
            "section": c.section,
            "page_start": c.page_start,
            "page_end": c.page_end,
        }
        for c in chunks
    ]
    documents = [c.text for c in chunks]
    collection.upsert(
        ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
    )


def delete_paper(paper_name: str) -> None:
    """Removes every stored chunk for a paper — call before re-embedding
    a changed paper, so a shrunk chunk count doesn't leave stale entries
    behind."""
    collection = _get_collection()
    collection.delete(where={"paper_name": paper_name})


def query(
    query_text: str, n_results: int = 5, where: dict[str, Any] | None = None
) -> dict:
    """
    Semantic search. `where` follows Chroma's filter syntax, e.g.
    {"section": "methodology"} or {"paper_name": "chebrolu2017ijrr.pdf"}.
    Boilerplate/administrative text is filtered from results before
    they're returned — see module docstring.
    """
    collection = _get_collection()
    query_embedding = embed_query(query_text)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * _OVERFETCH_MULTIPLIER,
        where=where,
    )

    ids, docs, metas, dists = [], [], [], []
    for id_, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        if _is_boilerplate(doc):
            continue
        ids.append(id_)
        docs.append(doc)
        metas.append(meta)
        dists.append(dist)
        if len(ids) >= n_results:
            break

    return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}


def count() -> int:
    return _get_collection().count()
