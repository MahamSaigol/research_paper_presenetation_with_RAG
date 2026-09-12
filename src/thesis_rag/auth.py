"""
Authentication against Microsoft Graph via MSAL's Device Code Flow.

Why device code flow (vs. Authorization Code Flow):
    Auth Code Flow is built for apps with a browser you control the redirect
    for (web apps, or desktop apps that can spin up a localhost listener and
    catch a redirect). A CLI ingestion script has neither. Device Code Flow
    is the flow Microsoft's own docs recommend for "input-constrained"
    clients: we print a URL + short code, you approve it in *any* browser on
    *any* device, and this process polls until you do. No redirect URI to
    register or listener to run.

Why a persisted, serializable token cache:
    Without it, every run of the ingestion script would force you through
    the device-code login again. MSAL's SerializableTokenCache holds
    access + refresh tokens; we dump it to disk after every acquisition
    attempt and reload it on startup. `acquire_token_silent` then transparently
    uses the cached refresh token to get a new access token when the old one
    expires (access tokens are short-lived, ~1hr) — you only see the login
    prompt again once the refresh token itself expires or is revoked
    (typically 90 days of inactivity).

Security note: `.token_cache.bin` contains live credentials to your
OneDrive. It's in .gitignore below — never commit it. For anything beyond a
personal portfolio project, encrypt it at rest (msal's cache_data is plain
JSON under the hood); out of scope for Phase 1 but worth flagging.
"""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

import msal

from .config import settings


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    cache_path = Path(settings.token_cache_path)
    if cache_path.exists():
        cache.deserialize(cache_path.read_text())
    return cache


def _persist_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        Path(settings.token_cache_path).write_text(cache.serialize())


def get_access_token() -> str:
    """
    Returns a valid Graph API access token, silently refreshing or
    prompting via device code flow as needed. This is the only function
    other modules should call — it hides the silent-vs-interactive branch.
    """
    cache = _load_cache()
    app = msal.PublicClientApplication(
        client_id=settings.graph_client_id,
        authority=settings.authority,
        token_cache=cache,
    )
    atexit.register(_persist_cache, cache)

    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(settings.scopes_list, account=accounts[0])

    if not result:
        flow = app.initiate_device_flow(scopes=settings.scopes_list)
        if "user_code" not in flow:
            raise RuntimeError(
                f"Failed to create device flow. Server response: {flow}"
            )
        print(flow["message"], file=sys.stderr)
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise RuntimeError(
            "Authentication failed: "
            f"{result.get('error')}: {result.get('error_description')}"
        )

    _persist_cache(cache)
    return result["access_token"]


if __name__ == "__main__":
    token = get_access_token()
    print(f"Got access token: {token[:20]}... (truncated)")
