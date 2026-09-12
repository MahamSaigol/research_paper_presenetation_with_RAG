"""
Splits extracted paper text into section-tagged, token-budgeted chunks.

Design choice: section boundaries come from extract.py's is_heading flag
(visual formatting), and section *meaning* comes from classify.py (an LLM
call), not from a keyword list in this module.
    An earlier version tried to canonicalize heading text directly here via
    regex ("Methods" -> methodology, etc.). That doesn't scale to 150
    papers with genuinely different heading vocabularies and structures
    (see extract.py's docstring for the concrete evidence). This module's
    job is now just mechanical: given sections that already have their
    final canonical label attached, cut each one into token-budgeted
    chunks with overlap.

Design choice: line-level chunk accumulation with a word-count-based token
*estimate*, not exact per-model tokenization (e.g. via tiktoken).
    We don't know your final embedding model yet (Phase 3), and chunk
    boundaries don't need to be exact — they need to be "roughly a few
    hundred tokens, with some overlap for context continuity across the
    cut". Exact tokenization would also mean depending on a specific
    tokenizer's vocab file, which for tiktoken means a network download on
    first use — an unnecessary fragility for what only needs to be a rough
    sizing signal. English text averages ~1.3 BPE tokens per word across
    most modern tokenizers, so `word_count * 1.3` is a good enough proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .extract import PageLine

_TOKENS_PER_WORD = 1.3


@dataclass
class Section:
    raw_heading: str  # the literal heading text as it appears in the PDF;
    # "" for the front-matter section before the first detected heading
    lines: list[PageLine]


def split_into_sections(lines: list[PageLine]) -> list[Section]:
    """
    Splits on extract.py's is_heading flag. Purely mechanical — no
    judgment about what a heading means happens here.
    """
    sections: list[Section] = []
    current = Section(raw_heading="", lines=[])

    for pl in lines:
        if pl.is_heading:
            if current.lines:
                sections.append(current)
            current = Section(raw_heading=pl.text, lines=[])
            continue  # the heading line itself isn't added as content
        current.lines.append(pl)

    if current.lines:
        sections.append(current)
    return sections


@dataclass
class Chunk:
    paper_name: str
    section: str  # canonical label, e.g. "methodology" — filled in by
    # classify.py's mapping before this is constructed
    raw_heading: str  # original heading text, kept for traceability/debugging
    chunk_index: int
    page_start: int
    page_end: int
    text: str

    def token_count(self) -> int:
        return _token_len(self.text)


def _token_len(text: str) -> int:
    return int(len(text.split()) * _TOKENS_PER_WORD)


def chunk_sections(
    paper_name: str, sections: list[Section], label_map: dict[str, str]
) -> list[Chunk]:
    """
    label_map maps each Section.raw_heading to its canonical category
    (from classify.py). The front-matter section (raw_heading == "") is
    always labeled "front_matter" directly, without needing classification
    — there's nothing to classify, it's just whatever precedes the first
    detected heading.
    """
    chunks: list[Chunk] = []
    max_tokens = settings.chunk_max_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    for section in sections:
        canonical = (
            "front_matter"
            if section.raw_heading == ""
            else label_map.get(section.raw_heading, "other")
        )
        buffer: list[PageLine] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n".join(pl.text for pl in buffer)
            chunks.append(
                Chunk(
                    paper_name=paper_name,
                    section=canonical,
                    raw_heading=section.raw_heading,
                    chunk_index=len(chunks),
                    page_start=min(pl.page_number for pl in buffer),
                    page_end=max(pl.page_number for pl in buffer),
                    text=text,
                )
            )

        for pl in section.lines:
            line_tokens = _token_len(pl.text)
            if buffer and buffer_tokens + line_tokens > max_tokens:
                flush()
                overlap_lines: list[PageLine] = []
                overlap_count = 0
                for prev in reversed(buffer):
                    t = _token_len(prev.text)
                    if overlap_count + t > overlap_tokens:
                        break
                    overlap_lines.insert(0, prev)
                    overlap_count += t
                buffer = overlap_lines
                buffer_tokens = overlap_count

            buffer.append(pl)
            buffer_tokens += line_tokens

        flush()

    return chunks
