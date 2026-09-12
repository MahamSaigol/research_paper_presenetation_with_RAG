"""
Phase 4: answers questions strictly from retrieved chunks, with citations
back to source paper + page, and explicit refusal when the corpus
doesn't actually contain an answer.

Design choice: two independent safety nets against ungrounded answers —
a distance-threshold pre-filter AND an explicit refusal instruction in
the prompt — not just one.
    Relying on either alone has a failure mode. A distance threshold
    alone is a blunt instrument: it can't distinguish "genuinely
    unrelated" from "borderline relevant, and the model should stay
    cautious about it" — it just catches the clearest cases. A prompt
    instruction alone can't stop retrieval from confidently returning
    *something* even when nothing in the corpus actually answers the
    question — top-k retrieval always returns k results, however weak
    the match, since Chroma has no built-in "there's nothing good
    enough" concept. So the LLM needs to see that risk named explicitly
    and be told refusing is a correct, expected answer, not a failure —
    which matters even more given a 3B model is less reliable at
    self-policing than a larger one would be.

Design choice: front_matter and references sections are excluded from
retrieval, before the LLM ever sees them.
    Confirmed against a real test query: a paper's front-matter chunk
    (DOI, journal name, publication date, restating the title) can rank
    highly on pure lexical/semantic overlap with a query without
    containing any actual descriptive content worth citing as evidence.
    references chunks are bibliography entries, not paper content.
    Filtering both out also frees the top-k slots for substantive chunks.

Design choice: relevance_distance_threshold (config) is a starting
estimate, not an empirically tuned value.
    Chroma's default distance metric is squared L2 over normalized
    embeddings, which relates to cosine similarity but isn't itself a
    familiar 0-1 scale. 1.0 is a reasonable starting cutoff, but the
    right value depends on your actual queries and corpus — if real
    questions are getting refused that shouldn't be, or answered from
    weak matches that shouldn't be, that's the value to adjust first,
    in .env, no code change needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .ollama_client import OllamaConnectionError, chat
from .vectorstore import query

_EXCLUDED_SECTIONS = {"front_matter", "references"}

_SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources \
provided below — never your own general knowledge, even if you already \
know the answer. Every claim in your answer must be supported by at \
least one source, cited inline as [N] matching that source's number.

If the sources don't actually contain enough information to answer the \
question — even if they're topically related — say so plainly: "The \
ingested papers don't contain enough information to answer this." Do \
not guess, extrapolate beyond what a source states, or fill gaps with \
outside knowledge. Refusing is the correct answer when the sources \
don't support one, not a failure."""


@dataclass
class Source:
    index: int
    paper_name: str
    section: str
    page_start: int
    page_end: int
    text: str
    distance: float

    def citation(self) -> str:
        pages = (
            f"p{self.page_start}"
            if self.page_start == self.page_end
            else f"p{self.page_start}-{self.page_end}"
        )
        return f"{self.paper_name}, {pages}"


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[Source]
    refused: bool  # True whenever no answer was generated (no relevant
    # chunks, or the model itself declined) — sources may still be
    # populated even when refused, so you can see what was *close* but
    # not good enough.


def _retrieve(question: str, k: int) -> list[Source]:
    # Over-fetch, since some of what comes back gets filtered out by
    # excluded section — otherwise a query could return fewer than k
    # usable sources even when better ones exist further down the list.
    results = query(question, n_results=k * 3)
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    sources: list[Source] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if meta["section"] in _EXCLUDED_SECTIONS:
            continue
        sources.append(
            Source(
                index=len(sources) + 1,
                paper_name=meta["paper_name"],
                section=meta["section"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                text=doc,
                distance=dist,
            )
        )
        if len(sources) >= k:
            break
    return sources


def _format_sources_for_prompt(sources: list[Source]) -> str:
    blocks = [f"[{s.index}] {s.citation()} ({s.section})\n{s.text}" for s in sources]
    return "\n\n".join(blocks)


def answer_question(question: str, k: int | None = None) -> AnswerResult:
    k = k or settings.retrieval_k
    sources = _retrieve(question, k)

    if not sources or sources[0].distance > settings.relevance_distance_threshold:
        return AnswerResult(
            question=question,
            answer="The ingested papers don't contain enough information to answer this.",
            sources=sources,
            refused=True,
        )

    user_prompt = f"Sources:\n{_format_sources_for_prompt(sources)}\n\nQuestion: {question}"

    try:
        raw = chat(
            _SYSTEM_PROMPT,
            user_prompt,
            model=settings.answer_model,
            timeout=settings.answer_timeout_seconds,
        )
    except OllamaConnectionError as e:
        return AnswerResult(
            question=question, answer=f"Error: {e}", sources=sources, refused=True
        )

    if raw is None:
        return AnswerResult(
            question=question,
            answer=(
                "The request to the local model timed out "
                f"({settings.answer_timeout_seconds}s). Try again, or a "
                "more specific question."
            ),
            sources=sources,
            refused=True,
        )

    answer_text = raw.strip()
    # The model was instructed to say this exact sentence when it can't
    # answer from the sources — detect that so callers (e.g. the CLI)
    # can distinguish a real answer from a model-initiated refusal
    # without re-parsing the whole prompt/response cycle.
    refused = "don't contain enough information" in answer_text.lower()

    return AnswerResult(question=question, answer=answer_text, sources=sources, refused=refused)
