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


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
