"""
Maps raw heading text (as it literally appears in a PDF — "Summary", "The
Agricultural Robot Platform: BoniRob", "3.1 Field Setup and Acquisition
Method") to one of a fixed set of canonical categories, using a local LLM
via Ollama.

Why an LLM here rather than more regex:
    extract.py finds heading *boundaries* by visual formatting, which
    generalizes well. But what a heading *means* genuinely requires
    understanding language, not pattern-matching words — "Summary" means
    the same thing as "Conclusion" in context, "The Agricultural Robot
    Platform: BoniRob" is describing the experimental setup, and some
    papers (dataset-description papers, in particular) simply don't have
    a Results or Limitations section at all, which a keyword list can't
    know versus just failing to match. This is exactly the kind of
    judgment call small LLMs are reliable at and hand-written pattern
    lists aren't.

Why batched in groups of 8, not one call for all of a paper's headings:
    A long paper can have 20-30+ headings. Tested against a real 34-page
    arXiv paper with that many, a single call both timed out (60s wasn't
    enough) and produced worse classifications — a smaller list is easier
    for a 3B model to get right in one pass. Groups of 8 keep each call
    fast and reliable; a paper with 25 headings costs 4 calls instead of 1,
    which is still cheap at ~150 papers total.

Why a Timeout is handled differently from a ConnectionError:
    A ConnectionError means Ollama isn't running at all — every paper in
    the run would fail identically, so it's raised immediately as a setup
    problem worth fixing before continuing. A Timeout means Ollama IS
    running but this particular batch took too long — that's local to one
    batch, not a sign the whole run is broken, so it's handled like a
    malformed response: log a warning, fall back to "other" for that
    batch's headings, and keep going with the rest of the paper.
"""

from __future__ import annotations

import json
import sys

from .config import settings
from .ollama_client import chat

CANONICAL_LABELS = {
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experimental_setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    "other",
}

_BATCH_SIZE = 8
_REQUEST_TIMEOUT_SECONDS = 120

_SYSTEM_PROMPT = f"""You classify academic paper section headings into a \
fixed set of categories. Categories: {", ".join(sorted(CANONICAL_LABELS))}.

Rules:
- Use "other" for anything that isn't really a paper section (titles, \
author names, keyword lists, table/figure captions that slipped through).
- Headings don't need to literally contain the category word — use \
meaning. "Summary" -> conclusion. "Approach" or "System Design" -> \
methodology. "Sensor Setup" or "Implementation Details" or "Dataset" -> \
experimental_setup. A heading describing a system/platform/pipeline the \
authors built -> methodology.
- Not every paper has every category — that's fine, just don't force a \
match that isn't really there; use "other" instead.
- A heading that names a system, platform, robot, or pipeline the authors \
built (e.g. "The Agricultural Robot Platform: BoniRob", "TrackerNet") is \
methodology, even if it doesn't contain a methodology-sounding word — the \
heading is naming a thing they built, which is what a methodology \
section describes.
- Numbered subsection headings (e.g. "3.2.1 Vision-Language Backbone" \
under a "3. Method" heading) should get their own best-fit category based \
on their own content/meaning, same as any other heading.
- Strong word cues, even if the heading isn't an exact category name: \
any heading containing "Result(s)" -> results. Any heading containing \
"Experiment(s)", "Experimental", "Setup", or "Dataset" -> \
experimental_setup. Any heading containing "Implementation", "Proposal", \
"Algorithm", "Approach", "Method(s)", "System", "Framework", "Model", or \
"Architecture" -> methodology. Trust these word cues over hesitation — \
"Experiments with remote servers" -> experimental_setup, "Research \
Proposal" -> methodology, "Implementation" -> experimental_setup, even \
though none of these exactly repeats a category name.
- You will be given a NUMBERED list of headings. Respond with ONLY a \
JSON object mapping each number (as a string) to one category string — \
e.g. {{"1": "introduction", "2": "related_work"}}. Use the numbers given, \
NOT the heading text, as keys. No other text."""


def _classify_batch(headings: list[str]) -> dict[str, str]:
    """Classifies a single batch (<= _BATCH_SIZE headings) in one call."""
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headings))
    user_prompt = f"Headings to classify:\n{numbered}"

    raw_content = chat(
        _SYSTEM_PROMPT,
        user_prompt,
        model=settings.classify_model,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        json_mode=True,
    )
    if raw_content is None:
        print(
            f"Warning: classification request timed out after "
            f"{_REQUEST_TIMEOUT_SECONDS}s for a batch of {len(headings)} "
            "headings. Falling back to 'other' for this batch.",
            file=sys.stderr,
        )
        return {h: "other" for h in headings}

    try:
        parsed = json.loads(raw_content)
        if not isinstance(parsed, dict):
            raise ValueError("Response JSON was not an object")
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"Warning: couldn't parse classification response ({e}). "
            "Falling back to 'other' for this batch.",
            file=sys.stderr,
        )
        return {h: "other" for h in headings}

    result: dict[str, str] = {}
    for i, heading in enumerate(headings):
        label = parsed.get(str(i + 1), "other")
        result[heading] = label if label in CANONICAL_LABELS else "other"
    return result


def classify_headings(raw_headings: list[str]) -> dict[str, str]:
    """
    Returns {raw_heading: canonical_label} for every heading in
    raw_headings (deduplicated), classified in batches of _BATCH_SIZE.
    A ConnectionError (Ollama not running) aborts immediately — that's a
    setup problem affecting every paper equally. A timeout or malformed
    response affects only its own batch and falls back to "other" for
    just those headings, so one slow/bad batch doesn't lose an entire
    paper's classification.
    """
    unique_headings = list(dict.fromkeys(raw_headings))  # de-dup, keep order
    if not unique_headings:
        return {}

    result: dict[str, str] = {}
    for start in range(0, len(unique_headings), _BATCH_SIZE):
        batch = unique_headings[start : start + _BATCH_SIZE]
        result.update(_classify_batch(batch))
    return result


if __name__ == "__main__":
    # Smoke test against a small fixed set of headings — run this first to
    # confirm Ollama + the model are reachable before running full ingest.
    test_headings = [
        "Introduction",
        "The Agricultural Robot Platform: BoniRob",
        "Summary",
        "References",
        "Keywords",
    ]
    result = classify_headings(test_headings)
    for heading, label in result.items():
        print(f"{heading!r} -> {label}")
