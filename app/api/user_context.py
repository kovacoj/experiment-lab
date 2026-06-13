"""Operator-supplied context store.

Two responsibilities:

1. Append-only persistence of free-text notes the operator attaches to
   a session (via POST /sessions/{id}/context or the chat widget's
   "log as evidence" button).

2. A small surface helper that summarizes recently logged notes so
   /refresh can attach them to its response metadata as a
   *supplementary* low-confidence signal. The labs themselves never
   consume these notes — they're operator memory, not training data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.api.storage import read_json, user_context_path, write_json

# Cap the log to keep the file bounded; oldest entries drop off first.
_MAX_ENTRIES = 500

# Default window for the recent-notes summary surfaced in /refresh.
_RECENT_WINDOW = 5


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load(session_id: str) -> list[dict[str, Any]]:
    payload = read_json(user_context_path(session_id))
    if not payload:
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return entries


def _save(session_id: str, entries: list[dict[str, Any]]) -> None:
    write_json(user_context_path(session_id), {"entries": entries[-_MAX_ENTRIES:]})


def append_entry(
    session_id: str,
    *,
    message: str,
    source: str = "manual",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a new entry and return the stored record (with id+timestamp)."""
    entry = {
        "entry_id": uuid4().hex[:12],
        "message": message.strip(),
        "source": source.strip() or "manual",
        "tags": list(tags or []),
        "created_at": _utcnow_iso(),
    }
    entries = _load(session_id)
    entries.append(entry)
    _save(session_id, entries)
    return entry


def list_entries(session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    entries = _load(session_id)
    if limit and len(entries) > limit:
        return entries[-limit:]
    return entries


def total_count(session_id: str) -> int:
    return len(_load(session_id))


def recent_context_summary(session_id: str, *, window: int = _RECENT_WINDOW) -> dict[str, Any] | None:
    """Summarize the last N entries for downstream consumers.

    Returns ``None`` when no entries exist (so /refresh can keep the
    user_context block absent rather than carrying empty noise).
    """
    entries = _load(session_id)
    if not entries:
        return None
    recent = entries[-window:]
    return {
        "count_recent": len(recent),
        "count_total": len(entries),
        "latest_at": recent[-1]["created_at"],
        "entries": [
            {
                "entry_id": e["entry_id"],
                "message": e["message"],
                "source": e["source"],
                "tags": e.get("tags", []),
                "created_at": e["created_at"],
            }
            for e in recent
        ],
        # Operator notes are explicitly low-confidence; this label is the
        # contract the dashboard / chat widget reads to badge them.
        "confidence_band": "operator-low",
    }
