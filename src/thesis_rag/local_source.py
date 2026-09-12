"""
Local-folder ingestion source (default mode).

Design choice: read directly from LOCAL_PAPERS_DIR, no copying/staging step.
    Since the folder is already local (synced manually, via OneDrive's
    desktop client, via rclone, or just dropped there by hand), there's
    nothing to "fetch" — list_local_papers() IS the fetch. This keeps the
    local-folder path deliberately the simplest possible implementation of
    the PaperFile interface, which is also why it's a good default: fewer
    moving parts between you and Phase 2.

Only *.pdf is listed. If your papers folder has other file types mixed in
(notes, exported bibtex, etc.), they're silently skipped rather than
erroring — this is a literature-review folder, not a strict single-purpose
directory, so being lenient here is the right default.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .paper_file import PaperFile


def list_local_papers(folder: str | None = None) -> list[PaperFile]:
    folder_path = Path(folder or settings.local_papers_dir).expanduser().resolve()

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Papers folder not found: {folder_path}\n"
            "Check LOCAL_PAPERS_DIR in .env — it should point at the folder "
            "containing your PDFs (e.g. after downloading/extracting them "
            "from OneDrive's web UI, or wherever OneDrive sync placed them)."
        )

    papers: list[PaperFile] = []
    for pdf_path in sorted(folder_path.glob("*.pdf")):
        stat = pdf_path.stat()
        papers.append(
            PaperFile(
                name=pdf_path.name,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                local_path=pdf_path,
            )
        )
    return papers


if __name__ == "__main__":
    # Smoke test: `python -m thesis_rag.local_source` lists what's in the
    # configured folder without touching Graph API at all.
    for p in list_local_papers():
        print(f"{p.name}  ({p.size / 1024:.1f} KB, modified {p.last_modified})")
