"""
NOTE: This is the optional Graph API ingestion mode (INGESTION_SOURCE=graph_api
in .env). The default mode is local_source.py — see that module and the
README for why. This module is left fully implemented and working for
anyone on an account/tenant where Azure AD app registration isn't blocked;
switching INGESTION_SOURCE=graph_api activates it with no other code changes.

Thin wrapper over the Graph API endpoints needed for Phase 1: listing and
downloading files from a specific OneDrive folder.

Endpoint choice: /me/drive/root:/{path}:/children
    Graph offers two ways to address a folder: by numeric drive-item ID, or
    by path relative to the drive root (the ":/path:" syntax). We use path
    addressing because it matches how you think about the folder ("Thesis/
    Literature") and needs no separate lookup step to resolve a path to an
    ID first. Trade-off: path segments with certain special characters need
    URL-encoding, handled below via `quote()`.

Download choice: the pre-authenticated @microsoft.graph.downloadUrl
    Each file's metadata includes a temporary, pre-signed download URL valid
    for a few minutes. Fetching that URL needs no Authorization header at
    all. This avoids a second call to /content that would otherwise be
    needed, and sidesteps a subtlety where /content sometimes issues a
    redirect your HTTP client has to follow with the auth header stripped.

Retry/backoff:
    Graph throttles aggressively under load and returns 429 with a
    Retry-After header. A portfolio project that silently fails under
    throttling looks unfinished; this retries a bounded number of times
    respecting that header. 5xx errors get the same treatment since they're
    usually transient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests

from .auth import get_access_token
from .config import settings
from .paper_file import PaperFile

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
MAX_RETRIES = 5


@dataclass
class DriveFile:
    id: str
    name: str
    size: int
    web_url: str
    download_url: str | None
    last_modified: str


class GraphClient:
    def __init__(self) -> None:
        self._session = requests.Session()

    def _headers(self) -> dict:
        # Fetched per-call rather than cached on the instance: get_access_token()
        # already handles silent refresh, so this keeps GraphClient stateless
        # with respect to token lifetime instead of duplicating that logic.
        return {"Authorization": f"Bearer {get_access_token()}"}

    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(MAX_RETRIES):
            resp = self._session.get(url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()  # last attempt's error, surfaced properly
        return resp

    def list_folder(self, folder_path: str | None = None) -> list[DriveFile]:
        """
        Lists files directly inside `folder_path` (default: the configured
        papers folder). Does not recurse into subfolders — add that later
        if your papers end up organized into subfolders per topic.
        """
        folder_path = folder_path or settings.onedrive_papers_folder
        encoded_path = quote(folder_path)
        url = f"{GRAPH_BASE}/me/drive/root:/{encoded_path}:/children"

        files: list[DriveFile] = []
        while url:
            resp = self._get_with_retry(url, headers=self._headers())
            payload = resp.json()
            for item in payload.get("value", []):
                if "file" not in item:
                    continue  # skip subfolders; see docstring note above
                files.append(
                    DriveFile(
                        id=item["id"],
                        name=item["name"],
                        size=item["size"],
                        web_url=item["webUrl"],
                        download_url=item.get("@microsoft.graph.downloadUrl"),
                        last_modified=item["lastModifiedDateTime"],
                    )
                )
            # Graph paginates results; follow @odata.nextLink until absent.
            url = payload.get("@odata.nextLink")
        return files

    def download_file(self, file: DriveFile, dest_dir: Path) -> Path:
        """Downloads a single file to dest_dir, returns the local path."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.name

        if file.download_url:
            # Pre-authenticated URL: no Authorization header needed or sent.
            resp = self._get_with_retry(file.download_url)
        else:
            # Fallback for the rare case downloadUrl wasn't included in the
            # listing response — re-fetch metadata for a fresh one.
            meta = self._get_with_retry(
                f"{GRAPH_BASE}/me/drive/items/{file.id}", headers=self._headers()
            ).json()
            resp = self._get_with_retry(meta["@microsoft.graph.downloadUrl"])

        dest_path.write_bytes(resp.content)
        return dest_path


def sync_graph_papers(dest_dir: Path) -> list[PaperFile]:
    """
    Downloads every file in the configured OneDrive folder into dest_dir
    and returns them as PaperFile — the same shape local_source.py produces.
    This is the adapter that lets Phase 2+ treat graph_api and local_folder
    identically; it's the only function outside code should call when
    running in graph_api mode.
    """
    client = GraphClient()
    papers: list[PaperFile] = []
    for f in client.list_folder():
        local_path = client.download_file(f, dest_dir)
        papers.append(
            PaperFile(
                name=f.name,
                size=f.size,
                last_modified=f.last_modified,
                local_path=local_path,
            )
        )
    return papers


if __name__ == "__main__":
    # Smoke test: lists the configured papers folder and prints what's there,
    # without downloading anything yet.
    client = GraphClient()
    for f in client.list_folder():
        print(f"{f.name}  ({f.size / 1024:.1f} KB, modified {f.last_modified})")
