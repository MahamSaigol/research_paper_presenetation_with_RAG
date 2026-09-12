"""
Tracks which papers have already been processed, so re-running ingestion
only does work for new or changed files.

Design choice: manifest keyed by filename, hash used to detect *changes*
to that filename — not keyed by hash alone.
    Keying purely by content hash can tell you "have I seen these exact
    bytes before", but can't cleanly express "this paper's PDF was
    replaced with a corrected version" — that shows up as one entry
    disappearing and an unrelated-looking one appearing. Keying by
    filename treats "a paper" as the stable identity (matching how you
    actually think about your literature — Smith2024.pdf is a paper you
    track over time), and the hash tells us whether *that paper's* content
    changed since last run.

Three outcomes per run, per file:
    - name not in manifest              -> new, process it
    - name in manifest, hash differs     -> changed, reprocess it
    - name in manifest, hash matches     -> unchanged, skip entirely
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings
from .paper_file import PaperFile


@dataclass
class ManifestEntry:
    sha256: str
    last_modified: str
    chunk_count: int
    # Set after a successful Phase 3 embedding run — the sha256 of the
    # paper's content at the time its chunks were last embedded. Compared
    # against the current sha256 to decide whether re-embedding is needed
    # (e.g. the paper changed since it was last embedded, or it's never
    # been embedded at all). None means "not embedded yet".
    embedded_sha256: str | None = None


def _manifest_path() -> Path:
    return Path(settings.data_dir) / "manifest.json"


def load_manifest() -> dict[str, ManifestEntry]:
    path = _manifest_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: ManifestEntry(**entry) for name, entry in raw.items()}


def save_manifest(manifest: dict[str, ManifestEntry]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {name: asdict(entry) for name, entry in manifest.items()}
    path.write_text(json.dumps(serializable, indent=2))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass
class DiffResult:
    to_process: list[PaperFile]  # new or changed — needs (re)extraction
    unchanged: list[str]  # paper names, skipped entirely
    removed: list[str]  # in manifest but no longer present in the source


def diff_papers(
    papers: list[PaperFile], manifest: dict[str, ManifestEntry]
) -> DiffResult:
    to_process: list[PaperFile] = []
    unchanged: list[str] = []
    seen_names: set[str] = set()

    for paper in papers:
        seen_names.add(paper.name)
        current_hash = sha256_file(paper.local_path)
        prior = manifest.get(paper.name)
        if prior is None or prior.sha256 != current_hash:
            to_process.append(paper)
        else:
            unchanged.append(paper.name)

    removed = [name for name in manifest if name not in seen_names]
    return DiffResult(to_process=to_process, unchanged=unchanged, removed=removed)
