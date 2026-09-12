"""
The shape Phase 2 (chunking, hash-dedup, embedding) is written against,
regardless of where a file came from.

Design choice: a shared dataclass instead of Phase 2 importing DriveFile
or LocalPaper directly.
    Without this, chunking/dedup code would need an `if isinstance(...)`
    branch (or two near-duplicate code paths) every time it touched a file's
    metadata. Both ingestion sources instead adapt their native
    representation into this one shape, and everything downstream — dedup
    hashing, chunking, the eventual vector store — only ever sees
    `PaperFile`. Switching ingestion_source in .env changes zero lines in
    Phase 2+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaperFile:
    name: str
    size: int
    last_modified: str  # ISO-8601 string, so both sources can agree on a format
    local_path: Path  # where Phase 2 can actually read the bytes from
