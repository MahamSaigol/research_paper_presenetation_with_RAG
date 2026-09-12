"""
Phase 3 entry point: reads chunks written by Phase 2 (data/chunks/*.json)
and embeds them into the Chroma vector store — skipping papers whose
chunks haven't changed since they were last embedded.

Requires Phase 2 (scripts/ingest.py) to have already run — this script
only reads what's already on disk, it doesn't re-extract or re-classify
anything.

Usage:
    python scripts/embed.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.chunk import Chunk  # noqa: E402
from thesis_rag.config import settings  # noqa: E402
from thesis_rag.dedup import load_manifest, save_manifest  # noqa: E402
from thesis_rag.vectorstore import delete_paper, upsert_chunks  # noqa: E402


def _load_chunks(paper_name: str) -> list[Chunk]:
    stem = Path(paper_name).stem
    path = Path(settings.data_dir) / "chunks" / f"{stem}.json"
    raw = json.loads(path.read_text())
    return [Chunk(**c) for c in raw]


def main() -> None:
    manifest = load_manifest()
    if not manifest:
        print("No papers tracked yet — run scripts/ingest.py first.")
        return

    to_embed = [
        name
        for name, entry in manifest.items()
        if entry.embedded_sha256 != entry.sha256
    ]
    skipped = len(manifest) - len(to_embed)
    print(f"{skipped} paper(s) already embedded and unchanged, skipped.")
    print(f"{len(to_embed)} paper(s) to embed.\n")

    for paper_name in to_embed:
        chunks_path = Path(settings.data_dir) / "chunks" / f"{Path(paper_name).stem}.json"
        if not chunks_path.exists():
            print(f"  Skipping {paper_name}: no chunks file found (run ingest.py first).")
            continue

        print(f"Embedding {paper_name} ...")
        chunks = _load_chunks(paper_name)
        # Clear any stale chunks from a previous embedding of this paper
        # first, in case chunking changed the count/IDs since last time.
        delete_paper(paper_name)
        upsert_chunks(chunks)
        print(f"  {len(chunks)} chunks embedded.")

        manifest[paper_name].embedded_sha256 = manifest[paper_name].sha256

    save_manifest(manifest)
    print(f"\nDone. {len(to_embed)} paper(s) embedded this run.")


if __name__ == "__main__":
    main()
