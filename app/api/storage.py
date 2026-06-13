"""On-disk JSON storage for sessions.

All API-layer state — monitoring plans and per-chart history — lives under
`tmp/sessions/{session_id}/`. The base directory can be overridden with the
`EXPERIMENT_LAB_API_TMP_DIR` env var (tests use this to isolate runs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def base_dir() -> Path:
    override = os.environ.get("EXPERIMENT_LAB_API_TMP_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "tmp" / "sessions"


def session_dir(session_id: str) -> Path:
    return base_dir() / session_id


def history_dir(session_id: str) -> Path:
    return session_dir(session_id) / "history"


def monitoring_plan_path(session_id: str) -> Path:
    return session_dir(session_id) / "monitoring_plan.json"


def decision_cards_path(session_id: str) -> Path:
    """Latest decision cards from the most recent /refresh."""
    return session_dir(session_id) / "decision_cards.json"


def alerts_log_path(session_id: str) -> Path:
    """Append-only alert log. Each /refresh appends entries derived from the
    current lab results; /alerts reads back the tail."""
    return session_dir(session_id) / "alerts.json"


def user_context_path(session_id: str) -> Path:
    """Append-only log of operator-supplied context (manual evidence).

    Each entry is a free-text note tagged with a source label (e.g.
    "manual", "chat-log-evidence", "ops-handoff") plus a timestamp. The
    lab pipeline never trusts these as ground truth — they surface as
    low-confidence signals in refresh metadata so the dashboard can show
    "operator notes" alongside automated findings.
    """
    return session_dir(session_id) / "user_context.json"


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
