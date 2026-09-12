"""
One place that knows how to get PaperFiles regardless of INGESTION_SOURCE.

Design choice: this used to be duplicated inline in list_papers.py. Pulling
it out means ingest.py (Phase 2) and list_papers.py (Phase 1) share the
exact same source-dispatch logic instead of two copies drifting apart.
"""

from __future__ import annotations

from pathlib import Path

from .config import settings
from .paper_file import PaperFile


def get_all_papers() -> list[PaperFile]:
    if settings.ingestion_source == "local_folder":
        from .local_source import list_local_papers

        return list_local_papers()
    else:
        from .graph_client import sync_graph_papers

        staging_dir = Path(settings.data_dir) / "graph_downloads"
        return sync_graph_papers(staging_dir)
