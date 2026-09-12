"""
Shared low-level Ollama HTTP client, used by both classify.py (Phase 2
section classification) and answer.py (Phase 4 answer generation).

Design choice: extracted into its own module now, at the point Phase 4
becomes Ollama's second real caller.
    classify.py already had this exact HTTP-calling boilerplate (build
    request, distinguish ConnectionError from Timeout, parse the
    response). Duplicating it into answer.py would mean the same
    connection-error message and timeout behavior living in two places,
    drifting apart the next time either needs a fix. Pulling it out once
    there's a second real caller — rather than speculatively on day one —
    is the usual "wait for the second use case" rule for when to
    generalize shared code.

Design choice: an empty completion is logged loudly, not silently
returned as an empty string.
    Confirmed against a real run: llama3.1:8b occasionally returns a
    genuinely empty "content" field for a successful HTTP response — a
    known, if uncommon, Ollama/model quirk, distinct from a timeout or a
    real "nothing relevant" judgment. Silently returning "" would make
    this indistinguishable from the model correctly finding nothing to
    say, which is a completely different situation and shouldn't be
    debugged by guessing. Logging Ollama's own `done_reason` for the
    response (e.g. "length" vs "stop") gives a concrete next clue instead
    of another blind guess.
"""

from __future__ import annotations

import sys

import requests

from .config import settings


class OllamaConnectionError(RuntimeError):
    """Ollama isn't reachable at all — a setup problem, not a per-request one."""


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    timeout: int,
    json_mode: bool = False,
    temperature: float | None = None,
) -> str | None:
    """
    Returns the model's raw text content, or None if the request timed
    out — callers decide their own fallback for that case (classify.py
    falls back to "other" for the batch; answer.py falls back to a
    "try again" message). Raises OllamaConnectionError if Ollama isn't
    running at all, since every caller should treat that as a setup
    problem worth surfacing immediately rather than silently degrading.

    temperature: None uses Ollama's model default. Confirmed necessary
    against a real paper: the exact same prompt and source text produced
    genuinely different completions across separate calls (a clean
    refusal one time, a hedge-then-real-bullets response another) —
    ordinary LLM sampling randomness, not a bug, but it made debugging
    "did retrieval work" indistinguishable from "did the model just roll
    differently this time." Callers doing careful content judgment
    (slidegen.py) pass a low value to make that judgment far more
    consistent run-to-run.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    try:
        response = requests.post(
            f"{settings.ollama_host}/api/chat", json=payload, timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(
            f"Couldn't reach Ollama at {settings.ollama_host}. Is it "
            "running? Start it with `ollama serve`, and make sure the "
            f"model is pulled: `ollama pull {model}`."
        ) from e
    except requests.exceptions.Timeout:
        return None

    body = response.json()
    content = body["message"]["content"]

    if not content.strip():
        print(
            f"Warning: {model} returned an empty completion. "
            f"done_reason={body.get('done_reason')!r}, "
            f"eval_count={body.get('eval_count')!r}. "
            "This is a genuine empty response, not 'nothing relevant found' "
            "— if this recurs, it's worth reporting/investigating as an "
            "Ollama/model issue rather than a retrieval or prompt problem.",
            file=sys.stderr,
        )

    return content
