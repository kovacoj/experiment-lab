"""Apify wrapper with cached-fixture fallback.

Lifted from `hackathon/api/apify_client.py` and adapted so that the
experiment-lab backend can ingest external reviews without depending on
the parallel hackathon subsystem at runtime.

Three modes (env-controlled):

  APIFY_MODE=cached  -> always return the cached fixture (default; demo-stable)
  APIFY_MODE=live    -> call Apify; on failure, fall back to cached + log it
  APIFY_MODE=off     -> raise; useful for tests that need to assert no-call

This module never imports the heavyweight `apify-client` SDK unless
``mode=live`` AND a token is configured. In cached mode it has zero
external dependencies.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Cached fixture lives at <repo>/fixtures/apify_reputation_reviews.json so
# the labs-first contract still holds: when APIFY_MODE!=live, no network
# call happens and the deterministic fixture drives the demo.
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "fixtures" / "apify_reputation_reviews.json"


def mode() -> str:
    return os.environ.get("APIFY_MODE", "cached").lower()


def load_cached(max_items: int = 20) -> list[dict[str, Any]]:
    """Read the cached Apify response and clamp to ``max_items``.

    Raises FileNotFoundError if the fixture is missing — that's a setup
    bug rather than a runtime failure we want to swallow.
    """
    items = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"cached apify fixture at {FIXTURE_PATH} is not a JSON list")
    return items[:max_items]


def run_actor(
    actor_id: str = "compass/google-maps-reviews-scraper",
    *,
    max_items: int = 20,
    token: str | None = None,
) -> dict[str, Any]:
    """Run an Apify actor (or return cached items, depending on mode).

    Returns a dict shaped like::

        {
          "mode": "cached" | "live" | "fallback",
          "actor_id": "...",
          "actor_run_id": "...",
          "items": [...],
          "error": None | "message",
        }

    Never raises in cached/live modes — failures are recorded in the dict
    so the caller can log them and continue (graceful demo fallback).
    """
    m = mode()
    if m == "off":
        raise RuntimeError("APIFY_MODE=off; refusing to run Apify actor.")

    if m == "cached":
        items = load_cached(max_items=max_items)
        return {
            "mode": "cached",
            "actor_id": actor_id,
            "actor_run_id": "cached-fixture",
            "items": items,
            "error": None,
        }

    # live
    api_token = token or os.environ.get("APIFY_TOKEN")
    if not api_token:
        # No token; fall back rather than raise so the demo stays up.
        items = load_cached(max_items=max_items)
        return {
            "mode": "fallback",
            "actor_id": actor_id,
            "actor_run_id": "cached-fixture",
            "items": items,
            "error": "APIFY_TOKEN not set",
        }

    try:  # pragma: no cover — only exercised when a real token is present
        from apify_client import ApifyClient  # type: ignore

        client = ApifyClient(api_token)
        run = client.actor(actor_id).call(
            run_input={"maxItems": max_items},
            timeout_secs=120,
        )
        run_id = run["id"] if run else "unknown"
        dataset_id = run.get("defaultDatasetId") if run else None
        items: list[dict[str, Any]] = []
        if dataset_id:
            items = list(client.dataset(dataset_id).iterate_items())[:max_items]
        return {
            "mode": "live",
            "actor_id": actor_id,
            "actor_run_id": run_id,
            "items": items,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        items = load_cached(max_items=max_items)
        return {
            "mode": "fallback",
            "actor_id": actor_id,
            "actor_run_id": "cached-fixture",
            "items": items,
            "error": f"apify failure: {type(exc).__name__}: {exc}",
        }


def fetch_dataset_items(dataset_id: str, *, max_items: int = 20, token: str | None = None) -> dict[str, Any]:
    """Fetch items directly from an existing Apify dataset id.

    In cached mode this ignores ``dataset_id`` and returns the cached
    fixture (so demo flows that name a "live dataset" still work
    offline). In live mode it pulls the actual dataset rows.
    """
    m = mode()
    if m == "off":
        raise RuntimeError("APIFY_MODE=off; refusing to fetch Apify dataset.")

    if m == "cached":
        items = load_cached(max_items=max_items)
        return {
            "mode": "cached",
            "actor_id": None,
            "actor_run_id": f"cached-for:{dataset_id}",
            "items": items,
            "error": None,
        }

    api_token = token or os.environ.get("APIFY_TOKEN")
    if not api_token:
        items = load_cached(max_items=max_items)
        return {
            "mode": "fallback",
            "actor_id": None,
            "actor_run_id": f"cached-for:{dataset_id}",
            "items": items,
            "error": "APIFY_TOKEN not set",
        }

    try:  # pragma: no cover — only exercised with a real token
        from apify_client import ApifyClient  # type: ignore

        client = ApifyClient(api_token)
        items = list(client.dataset(dataset_id).iterate_items())[:max_items]
        return {
            "mode": "live",
            "actor_id": None,
            "actor_run_id": f"dataset:{dataset_id}",
            "items": items,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        items = load_cached(max_items=max_items)
        return {
            "mode": "fallback",
            "actor_id": None,
            "actor_run_id": f"cached-for:{dataset_id}",
            "items": items,
            "error": f"apify failure: {type(exc).__name__}: {exc}",
        }
