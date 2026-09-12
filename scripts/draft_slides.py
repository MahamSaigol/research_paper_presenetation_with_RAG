"""
Phase 6 entry point: drafts the 6 slides for a paper and saves them as a
.pptx you review and correct.

Requires Phase 3 (scripts/embed.py) to have already run for the
paper(s) you're drafting — this queries the vector store via semantic
search, it doesn't read Phase 2's chunk files directly.

Usage:
    python scripts/draft_slides.py "WeedNet-X"        # matches by substring
    python scripts/draft_slides.py --all              # every paper in the manifest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.config import settings  # noqa: E402
from thesis_rag.dedup import load_manifest  # noqa: E402
from thesis_rag.pptx_export import build_pptx  # noqa: E402
from thesis_rag.slidegen import draft_all_slides  # noqa: E402


def _draft_and_save(paper_name: str) -> None:
    print(f"Drafting slides for: {paper_name}")
    drafts = draft_all_slides(paper_name)

    for d in drafts:
        status = f"{len(d.bullets)} bullets" if d.bullets else "not stated"
        print(f"  {d.title}: {status}")

    out_dir = Path(settings.data_dir) / "slides"
    out_path = out_dir / f"{Path(paper_name).stem}.pptx"
    build_pptx(paper_name, drafts, out_path)
    print(f"  Saved: {out_path}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    manifest = load_manifest()
    if not manifest:
        print("No papers tracked yet — run scripts/ingest.py first.")
        return

    if sys.argv[1] == "--all":
        for paper_name in manifest:
            _draft_and_save(paper_name)
        return

    query_substr = " ".join(sys.argv[1:]).lower()
    matches = [name for name in manifest if query_substr in name.lower()]

    if not matches:
        print(f"No paper matched {query_substr!r}. Tracked papers:")
        for name in manifest:
            print(f"  {name}")
        return

    if len(matches) > 1:
        print(f"Multiple papers matched {query_substr!r} — be more specific:")
        for name in matches:
            print(f"  {name}")
        return

    _draft_and_save(matches[0])


if __name__ == "__main__":
    main()
