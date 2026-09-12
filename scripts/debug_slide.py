"""
Diagnostic for Phase 6: shows exactly what happened for one paper/slide
category — the raw retrieved candidates, their distances, which ones
passed the relevance filters, and what the LLM actually said. Use this
when a slide comes back "not clearly stated" and you're not sure whether
that's because retrieval found nothing relevant, or because the LLM saw
something and declined anyway.

Usage:
    python scripts/debug_slide.py "WeedNet-X" methodology
    (category is one of: problem, methodology, experimental_setup,
     results, contributions, limitations)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.dedup import load_manifest  # noqa: E402
from thesis_rag.slidegen import (  # noqa: E402
    _EXCLUDE_RELATED_WORK_FOR,
    _MAX_SOURCE_CHARS_WIDE,
    _RELEVANCE_DISTANCE_CUTOFF,
    _RELEVANCE_MARGIN_WIDE,
    _SYSTEM_PROMPT_TEMPLATE,
    _TOP_K_WIDE,
    SLIDE_CATEGORIES,
)
from thesis_rag.vectorstore import query as vector_query
from thesis_rag.ollama_client import chat
from thesis_rag.config import settings


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    query_substr = sys.argv[1].lower()
    category_key = sys.argv[2].lower()

    manifest = load_manifest()
    matches = [name for name in manifest if query_substr in name.lower()]
    if len(matches) != 1:
        print(f"Expected exactly one paper match, got {len(matches)}: {matches}")
        return
    paper_name = matches[0]

    cat = next((c for c in SLIDE_CATEGORIES if c[0] == category_key), None)
    if cat is None:
        print(f"Unknown category {category_key!r}. Options: {[c[0] for c in SLIDE_CATEGORIES]}")
        return
    _, title, search_query = cat

    # Retrieval breadth (top_k/margin/budget) is uniform across all
    # categories now — only the related_work exclusion stays
    # category-specific. See slidegen.py's _EXCLUDE_RELATED_WORK_FOR
    # comment for why.
    exclude_related_work = category_key in _EXCLUDE_RELATED_WORK_FOR
    top_k = _TOP_K_WIDE
    margin = _RELEVANCE_MARGIN_WIDE
    max_chars = _MAX_SOURCE_CHARS_WIDE

    print(f"Paper: {paper_name}")
    print(f"Slide: {title}")
    print(f"Query: {search_query!r}")
    print(f"Settings: top_k={top_k}, margin={margin}, exclude_related_work={exclude_related_work}\n")

    where = (
        {"$and": [{"paper_name": paper_name}, {"section": {"$ne": "related_work"}}]}
        if exclude_related_work
        else {"paper_name": paper_name}
    )
    results = vector_query(search_query, n_results=top_k, where=where)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not dists:
        print("RETRIEVAL: zero candidates returned at all (nothing embedded for this paper?)")
        return

    best_distance = min(dists)
    print(f"RETRIEVAL: {len(dists)} candidates, best distance={best_distance:.3f}, "
          f"absolute cutoff={_RELEVANCE_DISTANCE_CUTOFF}, relative margin={margin}\n")

    total_chars = 0
    kept_text_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        passes_absolute = dist <= _RELEVANCE_DISTANCE_CUTOFF
        passes_relative = dist <= best_distance + margin
        under_budget = total_chars < max_chars
        kept = passes_absolute and passes_relative and under_budget
        if kept:
            total_chars += len(doc)
            kept_text_parts.append(doc)
        status = "KEPT" if kept else "dropped"
        reason = "" if kept else (
            "(too far, absolute)" if not passes_absolute else
            "(too far, relative)" if not passes_relative else
            "(over char budget)"
        )
        print(f"  [{i+1}] dist={dist:.3f} section={meta['section']:20s} "
              f"p{meta['page_start']}-{meta['page_end']}  {status} {reason}")
        print(f"      {doc[:150].strip()}...")

    print()
    if not kept_text_parts:
        print("RESULT: no candidates survived filtering — LLM was never called. "
              "This means the fix is about retrieval/filtering, not the model.")
        return

    source_text = "\n\n".join(kept_text_parts)
    print(f"Sending {len(source_text)} chars to {settings.answer_model}...\n")

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        slide_title=title, slide_title_lower=title.lower(), not_stated="Not clearly stated in this paper."
    )
    raw = chat(system_prompt, f"Source text:\n{source_text}", model=settings.answer_model, timeout=180)
    print("LLM RESPONSE:")
    print(raw)


if __name__ == "__main__":
    main()
