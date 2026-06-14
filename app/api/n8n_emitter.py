"""Fire-and-forget mirror of refresh events to an n8n webhook.

When ``N8N_REFRESH_WEBHOOK_URL`` is set, every successful
``POST /sessions/{id}/refresh`` posts a minimal summary payload to the
n8n webhook so n8n's execution log becomes a unified audit trail
regardless of trigger source (browser, MCP, schedule).

Design notes:

- This is **best-effort** and runs via FastAPI's ``BackgroundTasks``
  *after* the response is sent — the browser must never wait on n8n.
- All exceptions are swallowed and logged; n8n being down must not
  break the dashboard.
- The payload is intentionally tiny (no raw lab output, no card
  bodies). n8n is for routing, not for storing analytics state.
- No authentication header is sent. The webhook is documented as
  local-demo-only; tighten later if the deployment surface changes.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.api.schemas import RefreshResponse

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.0
_ENV_KEY = "N8N_REFRESH_WEBHOOK_URL"


def _webhook_url() -> str | None:
    url = os.environ.get(_ENV_KEY, "").strip()
    return url or None


def _payload(session_id: str, source: str, response: RefreshResponse) -> dict[str, Any]:
    alert_title = response.alert.title if response.alert else None
    return {
        "session_id": session_id,
        "source": source,
        "scenario": response.scenario,
        "prediction_changed": bool(response.prediction_changed),
        "decision_card_count": len(response.decision_cards or []),
        "alert_title": alert_title,
    }


def emit_refresh_event(
    session_id: str,
    source: str,
    response: RefreshResponse,
) -> None:
    """Mirror a refresh response to the configured n8n webhook.

    Safe to call from ``BackgroundTasks``. Returns ``None`` whether the
    POST succeeds, fails, or is skipped because the env var is unset.
    """
    url = _webhook_url()
    if not url:
        return
    body = _payload(session_id, source, response)
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            client.post(url, json=body)
    except Exception as exc:  # noqa: BLE001 — emitter is best-effort
        logger.warning("n8n refresh-event mirror failed: %s", exc)
