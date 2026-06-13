"""Append-only JSON history of chart data points.

One file per (session_id, metric_key). Each refresh appends one record;
the chart data endpoint reads the file back as-is.

The metric_key is intentionally one-per-chart for the current slice so the
chart endpoint can return the file contents verbatim. If a future slice needs
per-series files, add a second helper rather than overloading this one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.api.storage import history_dir, read_json, write_json


def _path(session_id: str, metric_key: str):
    return history_dir(session_id) / f"{metric_key}.json"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_metric_point(session_id: str, metric_key: str, point: dict[str, Any]) -> None:
    path = _path(session_id, metric_key)
    existing = read_json(path) or {"metric_key": metric_key, "points": []}
    points: list[dict[str, Any]] = list(existing.get("points", []))
    points.append(point)
    write_json(path, {"metric_key": metric_key, "points": points})


def read_metric_series(session_id: str, metric_key: str, limit: int = 200) -> list[dict[str, Any]]:
    path = _path(session_id, metric_key)
    existing = read_json(path)
    if not existing:
        return []
    points = list(existing.get("points", []))
    if limit and len(points) > limit:
        return points[-limit:]
    return points
