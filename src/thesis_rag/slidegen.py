"""
Phase 6: drafts first-pass content for the 6 slides you manually create
per paper (Problem, Methodology, Experimental Setup, Results,
Contributions, Limitations) — grounded in that paper's own chunks, meant
to be corrected by you afterward, not treated as final.

Design choice: source content per slide comes from Phase 3's semantic
search (scoped to one paper), NOT from Phase 2's section classification
labels. This replaced an earlier, weaker design.
    The first version filtered chunks by their classify.py section label
    (e.g. only "methodology"-tagged chunks feed the Methodology slide).
    Tested against a real paper, this failed badly: Phase 2's classifier
    (a small local LLM) mislabeled several of this paper's sections, so
    Methodology, Experimental Setup, and Results all came back either
    "not stated" or — after a fallback that filled leftover budget from
    "other"-tagged chunks in document order — filled with the *same*
    generic page-1/2 introduction text, since that's whatever "other"
    content happened to appear earliest in the paper. No amount of
    prompt tuning or fallback tuning fixes a fundamentally weak source
    signal. Phase 3's embeddings are a much stronger tool for exactly
    this problem: rather than trusting a label, each slide asks a
    natural-language question ("What method or architecture does this
    paper propose?") and retrieves the chunks — from this paper only —
    that are actually semantically closest to it, regardless of what
    section they got classified into. This reuses infrastructure already
    validated in Phase 4 rather than depending on Phase 2's weaker link.

Design choice: grounded generation, same discipline as Phase 4 — if
retrieval doesn't surface anything that actually addresses a slide's
topic, the slide says so explicitly rather than the model inventing
something plausible.
    A fabricated limitation or contribution is exactly the kind of error
    that's easy to miss when skimming a slide deck rather than reading
    carefully.

Requires Phase 3 (scripts/embed.py) to have already run for a paper —
this queries the vector store, it doesn't read chunks off disk directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .ollama_client import chat
from .vectorstore import query as vector_query

# Display name, and the natural-language query used to retrieve source
# content for it via semantic search (scoped to one paper). This
# replaces the earlier section-label-based mapping — see module
# docstring for why.
#
# Queries for methodology/contributions/limitations are phrased to
# explicitly say "this paper's own X, not related/prior work" — tested
# against a real paper where retrieval otherwise pulled in Related
# Work's discussion of *other* methods' techniques/shortcomings, which
# reads as topically on-target to a generic query but is attributionally
# wrong (describing someone else's work, not this paper's own).
SLIDE_CATEGORIES: list[tuple[str, str, str]] = [
    ("problem", "Problem", "What problem, challenge, or motivation does this paper address?"),
    ("methodology", "Methodology", "What method, architecture, or algorithm does THIS paper itself propose or build (not prior or related work)?"),
    ("experimental_setup", "Experimental Setup", "What datasets, experimental setup, or training configuration were used to evaluate this?"),
    ("results", "Results", "What are the quantitative results, accuracy, or performance numbers reported?"),
    ("contributions", "Contributions", "What are the main contributions THIS paper itself claims to make (not what prior work did)?"),
    ("limitations", "Limitations", "What limitations, weaknesses, or future work does THIS paper state about its OWN model (not criticisms of other methods)?"),
]

# Categories where Related Work is excluded outright from retrieval —
# confirmed necessary against a real paper: Related Work sections
# describe *other* methods using nearly the same vocabulary a genuine
# Methodology/Contributions/Limitations section would use for *this*
# paper, so it kept winning retrieval it shouldn't. Problem is
# deliberately NOT in this set: related_work content correctly and
# usefully answered it in that same test (prior methods' shortcomings
# are legitimate problem-framing), so excluding it there would remove
# good content. (Retrieval *breadth* — top_k/margin/budget — used to be
# tied to this same set, on the theory that only these 3 needed a wider
# net. A second real paper disproved that: Problem's default narrow
# top_k missed that paper's actual introduction content too. Breadth is
# now uniform across all 6 categories; only the related_work exclusion
# stays category-specific, since that one's benefit is genuinely
# category-dependent, confirmed by real evidence in both directions.)
_EXCLUDE_RELATED_WORK_FOR = {"methodology", "contributions", "limitations"}

_TOP_K = 4  # no longer used directly — kept as a documented reference
# point for how narrow the original (too-narrow) default was.
_TOP_K_WIDE = 10
_MAX_SOURCE_CHARS = 3500  # no longer used directly — see _TOP_K note above
# Applied to all categories now (see _EXCLUDE_RELATED_WORK_FOR comment
# for why) — with _TOP_K_WIDE pulling in up to 10 candidates, the old
# narrower budget would only fit 2-3 of them anyway, undermining the
# point of retrieving more. The larger model (8B) has more headroom to
# actually make use of a longer, multi-candidate source block than the
# 3B model this was originally tuned against.
_MAX_SOURCE_CHARS_WIDE = 6000
_TIMEOUT_SECONDS = 120
_NOT_STATED = "Not clearly stated in this paper."
# Distance cutoffs (squared L2, normalized embeddings — same convention
# as Phase 4). Two checks, not one: an absolute cutoff alone would still
# admit weak-but-technically-under-threshold matches just to fill out
# top_k when nothing great exists for this paper/category. The relative
# margin keeps only chunks close to the *best* match actually found, so
# a slide's source footer reflects genuinely relevant content rather
# than "whatever ranked in the top few, however loosely related."
_RELEVANCE_DISTANCE_CUTOFF = 1.2
_RELEVANCE_MARGIN = 0.35  # no longer used directly — see _TOP_K note above
# Applied to all categories now — paired with _TOP_K_WIDE, this lets
# more borderline-distance candidates through to the LLM rather than
# the mechanical filter excluding them before generation even runs.
_RELEVANCE_MARGIN_WIDE = 0.6

_SYSTEM_PROMPT_TEMPLATE = """You draft the "{slide_title}" slide for an \
academic-paper review presentation. Using ONLY the source text provided \
below — never your own general knowledge or assumptions — write 3-5 \
short bullet points suitable for a slide (each under ~20 words, not full \
sentences copied verbatim).

The source text was retrieved by semantic search and may not actually \
discuss {slide_title_lower}, even though it's topically related to the \
paper. Judge the CONTENT, not just whether text was provided: if the \
source text doesn't actually discuss {slide_title_lower} at all, \
respond with exactly: "{not_stated}"

Be careful about ONE specific failure mode: text that summarizes what \
OTHER researchers built or found (e.g. "Smith et al. proposed X", a \
survey of prior techniques) is NOT this paper's own {slide_title_lower}, \
even if it's topically similar — that should get "{not_stated}" too. \
But do NOT apply this caution to text describing what THIS paper itself \
did, used, or built, even when it involves something that originated \
elsewhere — e.g. "we evaluated on the SugarBeets2016 dataset" or "we \
used SGD with batch size 16" IS this paper's own experimental content, \
not related work, even though the dataset/optimizer itself isn't novel. \
When the source text plainly describes this paper's own work, use it —  \
don't refuse out of excess caution.

Do not guess, pad with generic statements, or invent detail the source \
doesn't state. Respond with ONLY the bullet points (one per line, no \
numbering, no extra commentary) or the not-stated sentence — nothing else. \
Never combine both: if you decide there's real content worth reporting, \
give ONLY the bullets and do not also open with the not-stated sentence."""


@dataclass
class SlideDraft:
    key: str
    title: str
    bullets: list[str]  # empty list means "not stated"
    source_pages: list[str]  # e.g. ["p3", "p5-6"] — for your manual verification


def _gather_source_text(
    paper_name: str,
    search_query: str,
    top_k: int,
    exclude_related_work: bool,
    margin: float,
    max_chars: int,
) -> tuple[str, list[str]]:
    where: dict = (
        {"$and": [{"paper_name": paper_name}, {"section": {"$ne": "related_work"}}]}
        if exclude_related_work
        else {"paper_name": paper_name}
    )
    results = vector_query(search_query, n_results=top_k, where=where)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not dists:
        return "", []
    best_distance = min(dists)

    text_parts: list[str] = []
    pages: list[str] = []
    total_chars = 0
    for doc, meta, dist in zip(docs, metas, dists):
        if dist > _RELEVANCE_DISTANCE_CUTOFF or dist > best_distance + margin:
            continue
        if total_chars >= max_chars:
            break
        text_parts.append(doc)
        total_chars += len(doc)
        page = (
            f"p{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"p{meta['page_start']}-{meta['page_end']}"
        )
        if page not in pages:
            pages.append(page)
    return "\n\n".join(text_parts), pages


def _draft_one_slide(key: str, title: str, search_query: str, paper_name: str) -> SlideDraft:
    exclude_related_work = key in _EXCLUDE_RELATED_WORK_FOR
    # Wide retrieval settings for every category, not just the original
    # 3 — confirmed necessary against a second real paper, where
    # Problem's default narrow top_k missed that paper's actual
    # introduction/motivation content (it just didn't rank as close for
    # that paper's writing style as it had for the first paper this was
    # tuned against), leaving the model to work with tangentially
    # related content and produce a plausible-sounding but
    # miscategorized answer (limitations-shaped text on the Problem
    # slide). The mechanical retrieval filters shouldn't be the reason
    # correct content never reaches the LLM, for any category.
    top_k = _TOP_K_WIDE
    margin = _RELEVANCE_MARGIN_WIDE
    max_chars = _MAX_SOURCE_CHARS_WIDE
    source_text, pages = _gather_source_text(
        paper_name, search_query, top_k, exclude_related_work, margin, max_chars
    )

    if not source_text.strip():
        return SlideDraft(key=key, title=title, bullets=[], source_pages=[])

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        slide_title=title, slide_title_lower=title.lower(), not_stated=_NOT_STATED
    )
    user_prompt = f"Source text:\n{source_text}"

    raw = chat(
        system_prompt,
        user_prompt,
        model=settings.answer_model,
        timeout=_TIMEOUT_SECONDS,
        # Low temperature — confirmed necessary against a real paper,
        # where the same retrieval produced a clean refusal on one call
        # and a hedge-then-real-bullets response on another. This
        # judgment call should be as consistent as possible run-to-run.
        temperature=0.1,
    )

    if raw is None:
        return SlideDraft(
            key=key, title=title,
            bullets=["(Draft generation timed out — rerun this paper.)"],
            source_pages=pages,
        )

    raw = raw.strip()

    # Line-level filtering, not a whole-response substring check.
    # Confirmed necessary against a real response: the model sometimes
    # opens with the exact "not clearly stated" sentence and then
    # continues anyway with genuinely relevant bullets — a hedge-then-
    # answer pattern, not a clean refusal. A substring check ("does the
    # refusal phrase appear anywhere") would discard that entire
    # response, throwing away real content the model actually
    # generated. Only treat this as a genuine refusal if EVERY
    # non-empty line is (close to) the refusal sentence itself;
    # otherwise keep whichever lines aren't.
    not_stated_normalized = _NOT_STATED.rstrip(".").lower()
    all_lines = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    bullets = [
        line for line in all_lines
        if line.rstrip(".").lower() != not_stated_normalized
    ]

    if not bullets:
        return SlideDraft(key=key, title=title, bullets=[], source_pages=[])

    return SlideDraft(key=key, title=title, bullets=bullets, source_pages=pages)


def draft_all_slides(paper_name: str) -> list[SlideDraft]:
    """Drafts all 6 slides for one paper, one semantic search + one LLM call each."""
    return [
        _draft_one_slide(key, title, search_query, paper_name)
        for key, title, search_query in SLIDE_CATEGORIES
    ]
