"""
PDF text extraction, page by page, with two-column reading-order correction
and heading-candidate detection.

Design choice: PyMuPDF (fitz) over pypdf.
    Academic papers (yours are mostly CVPR/WACV/IEEE two-column layouts)
    break naive top-to-bottom text extraction: reading strictly by vertical
    position interleaves lines from both columns. PyMuPDF exposes text as
    positioned blocks (x0, y0, x1, y1, text), which lets us detect the
    column split and sort left-column-top-to-bottom, then right-column-top-
    to-bottom — much closer to actual reading order.
    Trade-off: PyMuPDF is AGPL-licensed. Fine for a personal/open-source
    repo; flag it if this code is ever reused inside closed-source software.

Design choice: heading candidates detected by VISUAL formatting (font size
relative to the document's body text, bold flag, or ALL-CAPS-with-Roman-
numeral-prefix), not by matching heading text against a keyword list.
    An earlier version tried to recognize headings by their words
    ("Methodology", "Results", etc.). Tested against real papers, this
    failed badly: one is Springer LNCS style where "Abstract." runs inline
    with the first sentence rather than sitting on its own line; another
    uses headings like "The Agricultural Robot Platform: BoniRob" and
    "Summary" that no keyword list would anticipate. Across 150 papers
    spanning many venues, a keyword list was never going to keep up.
    Visual formatting generalizes much better — but "visual formatting"
    turned out to mean more than one thing. Some IEEE-style templates
    signal headings purely through ALL-CAPS + a Roman-numeral prefix
    ("II. LITERATURE REVIEW"), with no size or bold difference from body
    text at all — confirmed against a real paper where "II. LITERATURE
    REVIEW" was 9.96pt, non-bold, against a 10pt body. That's why there
    are two independent heading signals below, not one.
    This module only finds heading *boundaries*; classify.py decides what
    each one actually means.

Design choice: suppressing long runs of consecutive heading candidates,
regardless of which signal (size, bold, or ALL-CAPS-Roman) triggered them.
    Tested against two real papers: one whose entire Abstract paragraph
    was set in bold-italic (every line looked like a bold heading
    candidate), another with 10+ co-authors each on their own line in a
    font larger than body text (every author name looked like a
    size-triggered heading candidate). A genuine heading is 1-2 lines; a
    long run of consecutive candidates is virtually always a styled
    paragraph or an author/title block, not a sequence of real headings.
    Runs longer than _MAX_HEADING_RUN are downgraded back to ordinary
    body text — which is the right outcome either way, since title/author
    blocks get classified "other" whether or not they're kept as
    headings, so suppressing them just makes the resulting front_matter
    section cleaner.

Filtering notes (see _is_noise / _NUMBERING / _REFERENCES_HEADING):
    Font-based detection alone also flags table captions, table header
    cells, and pieces of code/YAML listings that happen to be styled the
    same as a real heading. These are filtered by structural pattern
    (starts with "Table"/"Figure", ends with ":", etc.) rather than by
    content — the same "shape, not words" principle as heading detection
    itself. Boundary detection also deliberately stops at the first
    References/Bibliography heading: reference-list entries sometimes
    inherit heading-like styling from PDF generators, and nothing after
    References needs sub-sectioning in an academic paper anyway.
Design choice: ALL-CAPS-Roman-numeral matches are exempt from run
suppression and always break/reset the run.
    Tested against the same bold-italic-abstract paper: the abstract
    paragraph (correctly suppressed) sat directly adjacent to "I.
    INTRODUCTION" with no ordinary body-text line between them to reset
    the run counter, so "I. INTRODUCTION" was getting swept into the same
    oversized run and incorrectly suppressed along with it. The
    ALL-CAPS-Roman pattern is specific enough that a false match is
    very unlikely (unlike bold or oversized text, which legitimately
    appear in non-heading contexts like author lists), so it's treated as
    unconditionally real: it flushes whatever's currently buffered
    (applying the normal run-length check to that), then is appended
    directly as its own protected heading, never subject to suppression.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF; `fitz` is the historical import alias

_BOLD_FLAG = 1 << 4  # PyMuPDF span flags bit 4 = bold
_HEADING_SIZE_RATIO = 1.1  # heading font must be >10% larger than body text
_MAX_HEADING_LEN = 70
_MAX_HEADING_RUN = 2  # max consecutive heading candidates before treating as a block, not headings

_CAPTION_PREFIX = re.compile(r"^(table|figure|fig\.?|listing)\s", re.IGNORECASE)
_NUMBERING_ONLY = re.compile(r"^\s*(?:[ivxlc]+\.|\d+(?:\.\d+)*\.?)\s*$", re.IGNORECASE)
_NUMBERING_PREFIX = re.compile(r"^\s*(?:[ivxlc]+\.|\d+(?:\.\d+)*\.?)\s*", re.IGNORECASE)
_REFERENCES_HEADING = re.compile(r"^(references|bibliography)$", re.IGNORECASE)
_TABLE_WORDS = {"parameter", "value", "metric", "result", "data", "description"}

# "II. LITERATURE REVIEW", "III. RESEARCH PROPOSAL" — Roman numeral prefix
# followed by an all-uppercase title. Case-sensitive on purpose: the whole
# point is that the title portion has no lowercase letters at all.
_ALLCAPS_ROMAN_HEADING = re.compile(r"^[IVXLC]+\.\s+[A-Z][A-Z0-9 ,\-:&/]*$")


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return True
    if _CAPTION_PREFIX.match(stripped):
        return True
    if stripped.endswith(":"):
        # Covers short heading-style fragments from code/YAML listings
        # ("type:", "points:") and letter-spaced monospace artifacts
        # ("f i l e n a m e :") that PDF extraction sometimes produces.
        return True
    if stripped.rstrip(".,;").lower() in _TABLE_WORDS:
        return True
    return False


@dataclass
class PageLine:
    page_number: int  # 1-indexed, matches what a human would cite
    text: str
    is_heading: bool = False


def _detect_body_font_size(doc: fitz.Document) -> float:
    # The most common font size by character count is a robust proxy for
    # "this is what body paragraph text looks like" — headings, captions,
    # and other short elements don't have enough characters to dominate.
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _sort_blocks_reading_order(blocks: list[dict], page_width: float) -> list[dict]:
    """
    Two-column-aware reading order, correctly handling full-width elements
    (titles, abstracts, author blocks, figures/tables spanning both
    columns) that don't fit a simple left/right split.

    Approach: walk blocks top-to-bottom. A block is treated as "full
    width" if it straddles the page's horizontal midpoint (x0 before it,
    x1 after it) — genuine column text never does this, since it's
    confined to one half of the page by the column gutter, but centered
    single-column text (titles, abstracts) reliably does, even when its
    total width happens to be well under the page width. Narrow,
    non-straddling blocks are buffered into a left or right bucket by
    which side of the midpoint they're on. Each full-width block acts as
    a flush point: everything buffered so far is emitted — left column
    fully, then right column fully, matching true column-major reading
    order — followed by the full-width block itself, before buffering
    resumes.
    """
    midpoint = page_width / 2
    tolerance = 5.0  # points; avoids misclassifying blocks that just touch the line

    ordered: list[dict] = []
    left_buf: list[dict] = []
    right_buf: list[dict] = []

    def flush() -> None:
        ordered.extend(left_buf)
        ordered.extend(right_buf)
        left_buf.clear()
        right_buf.clear()

    for block in sorted(blocks, key=lambda b: b["bbox"][1]):
        x0, _, x1, _ = block["bbox"]
        straddles_midpoint = x0 < midpoint - tolerance and x1 > midpoint + tolerance
        if straddles_midpoint:
            flush()
            ordered.append(block)
        else:
            center_x = (x0 + x1) / 2
            (left_buf if center_x < midpoint else right_buf).append(block)

    flush()
    return ordered


def extract_pages(pdf_path: Path) -> list[PageLine]:
    """
    Returns the document's text as a flat list of PageLine (page_number,
    text, is_heading), in corrected two-column reading order, with heading
    candidates flagged by visual formatting.
    """
    lines: list[PageLine] = []
    heading_run: list[PageLine] = []  # consecutive heading candidates, held pending a run-length check

    def flush_heading_run() -> None:
        nonlocal heading_run
        if not heading_run:
            return
        if len(heading_run) > _MAX_HEADING_RUN:
            # A long run of consecutive candidates (whatever triggered
            # them) is a title/author block or a styled paragraph, not a
            # sequence of real headings — downgrade them all to body text.
            for pl in heading_run:
                pl.is_heading = False
        lines.extend(heading_run)
        heading_run = []

    with fitz.open(pdf_path) as doc:
        body_size = _detect_body_font_size(doc)
        hit_references = False

        for page_index, page in enumerate(doc):
            if hit_references:
                break
            page_number = page_index + 1
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
            ordered_blocks = _sort_blocks_reading_order(blocks, page.rect.width)

            pending_numbering: str | None = None
            for block in ordered_blocks:
                if hit_references:
                    break
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue

                    max_size = max(s["size"] for s in spans)
                    is_bold = any(s["flags"] & _BOLD_FLAG for s in spans)
                    is_strong_size = max_size > body_size * _HEADING_SIZE_RATIO
                    is_allcaps_roman = bool(_ALLCAPS_ROMAN_HEADING.match(text))
                    not_noise = not _is_noise(text)
                    within_len = len(text) <= _MAX_HEADING_LEN

                    looks_like_heading = (
                        within_len
                        and not_noise
                        and (is_strong_size or is_bold or is_allcaps_roman)
                    )

                    if looks_like_heading and _NUMBERING_ONLY.match(text):
                        # Bare "3" or "III" on its own line — hold it and
                        # merge with the heading text that follows.
                        pending_numbering = text
                        continue

                    if looks_like_heading and pending_numbering:
                        text = f"{pending_numbering} {text}"
                        pending_numbering = None
                    elif pending_numbering:
                        # The pending number wasn't followed by a heading
                        # after all (e.g. a numbered reference) — drop it
                        # rather than merge it into unrelated body text.
                        pending_numbering = None

                    pl = PageLine(
                        page_number=page_number, text=text, is_heading=looks_like_heading
                    )

                    if is_allcaps_roman:
                        # High-precision signal — never suppressed, and
                        # always breaks/resets whatever run was building
                        # so it can't be swept into a suppressed block.
                        flush_heading_run()
                        lines.append(pl)
                    elif looks_like_heading:
                        heading_run.append(pl)
                    else:
                        flush_heading_run()
                        lines.append(pl)

                    if looks_like_heading and _REFERENCES_HEADING.match(
                        _NUMBERING_PREFIX.sub("", text).strip()
                    ):
                        hit_references = True
                        break

    flush_heading_run()
    return lines


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m thesis_rag.extract <path-to-pdf>")
        sys.exit(1)
    for pl in extract_pages(Path(sys.argv[1])):
        marker = "H" if pl.is_heading else " "
        print(f"[p{pl.page_number}][{marker}] {pl.text}")
