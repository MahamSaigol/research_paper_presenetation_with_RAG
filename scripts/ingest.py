"""
Phase 2 entry point: for every new or changed paper —
  1. extract text with heading boundaries (extract.py)
  2. classify each heading into a canonical section via local LLM (classify.py)
  3. chunk each section, token-budgeted, with overlap (chunk.py)
  4. write chunks to data/chunks/<name>.json
Unchanged papers (per the dedup manifest) are skipped entirely.

Requires Ollama running locally with the configured model pulled:
    ollama serve
    ollama pull llama3.2:3b     (or whatever CLASSIFY_MODEL is set to)

Usage:
    python scripts/ingest.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from thesis_rag.chunk import chunk_sections, split_into_sections  # noqa: E402
from thesis_rag.classify import classify_headings  # noqa: E402
from thesis_rag.config import settings  # noqa: E402
from thesis_rag.dedup import (  # noqa: E402
    ManifestEntry,
    diff_papers,
    load_manifest,
    save_manifest,
    sha256_file,
)
from thesis_rag.extract import extract_pages  # noqa: E402
from thesis_rag.ingestion import get_all_papers  # noqa: E402


def _chunks_path(paper_name: str) -> Path:
    stem = Path(paper_name).stem
    return Path(settings.data_dir) / "chunks" / f"{stem}.json"


def main() -> None:
    papers = get_all_papers()
    manifest = load_manifest()
    diff = diff_papers(papers, manifest)

    print(f"{len(diff.unchanged)} unchanged, skipped.")
    if diff.removed:
        print(
            f"{len(diff.removed)} paper(s) in the manifest no longer found in "
            f"the source: {', '.join(diff.removed)} "
            "(their old chunks on disk are left as-is; delete manually if "
            "you want them gone)."
        )
    print(f"{len(diff.to_process)} new/changed paper(s) to process.\n")

    for paper in diff.to_process:
        print(f"Processing {paper.name} ...")
        lines = extract_pages(paper.local_path)
        sections = split_into_sections(lines)

        raw_headings = [s.raw_heading for s in sections if s.raw_heading]
        try:
            label_map = classify_headings(raw_headings)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping {paper.name} for now — rerun once Ollama is up.")
            continue  # don't update the manifest; this paper stays "new" for next run

        chunks = chunk_sections(paper.name, sections, label_map)

        out_path = _chunks_path(paper.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2))

        section_labels = sorted({c.section for c in chunks})
        print(
            f"  {len(chunks)} chunks, sections found: {', '.join(section_labels)}"
        )

        manifest[paper.name] = ManifestEntry(
            sha256=sha256_file(paper.local_path),
            last_modified=paper.last_modified,
            chunk_count=len(chunks),
        )

    save_manifest(manifest)
    print(f"\nManifest updated: {len(manifest)} paper(s) tracked total.")


if __name__ == "__main__":
    main()
