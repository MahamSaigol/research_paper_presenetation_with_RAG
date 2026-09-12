"""
Phase 1 smoke test: lists whatever papers are available, using whichever
ingestion source is configured in .env (INGESTION_SOURCE).

Usage:
    python scripts/list_papers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.config import settings  # noqa: E402
from thesis_rag.ingestion import get_all_papers  # noqa: E402


def main() -> None:
    papers = get_all_papers()

    if not papers:
        print(
            f"Connected ({settings.ingestion_source} mode), but no PDFs found. "
            "Check your folder path in .env."
        )
        return

    print(f"Found {len(papers)} file(s) via {settings.ingestion_source}:\n")
    for p in papers:
        print(f"  - {p.name} ({p.size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
