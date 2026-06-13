"""Derive deterministic alerts from a LabRunReport.

This is the API-layer equivalent of an alert policy: it reads the freshly
computed lab results plus the current sentiment snapshot and emits one
alert entry per result that crosses a severity rule. Output shape is
stable so the dashboard + n8n MCP tools can render it.

Severity rules (slice-level, intentionally simple):
  * `at_risk`  sentiment < 0.60                    -> critical
  * `watch`    0.60 <= sentiment < 0.75            -> warning
  * `healthy`  sentiment >= 0.75                   -> info (suppressed)

For non-location-scoped results we fall back to result.status:
  * `warning` -> warning
  * `selected` with `final_priority_score` >= 0.7  -> warning (action-ready)
  * else                                             -> info (suppressed)

`suppressed` rows are NOT emitted by `derive_alerts`. The dashboard
should treat "no alerts" as healthy state.
"""
from __future__ import annotations

import re
from typing import Any

from app.api.history_store import utcnow_iso
from app.labs.schemas import LabResult, LabRunReport

# ---------------------------------------------------------------------------

_LOCATION_PATTERN = re.compile(r"miners?\s+([a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]+)", re.IGNORECASE)


def _severity_from_score(score: float) -> str:
    if score < 0.60:
        return "critical"
    if score < 0.75:
        return "warning"
    return "info"


def _extract_location_token(result: LabResult) -> str | None:
    """Best-effort extraction of the location name from a lab result."""
    for rule in result.monitoring_rules:
        rule_dict = rule if isinstance(rule, dict) else (
            rule.model_dump() if hasattr(rule, "model_dump") else dict(rule)
        )
        entity = rule_dict.get("entity_name")
        if entity:
            return str(entity)
    match = _LOCATION_PATTERN.search(result.summary or "")
    if match:
        return f"Miners {match.group(1).capitalize()}"
    return None


def _slug_for(name: str) -> str:
    """Match `sentiment_metrics.slug` style: lowercase underscored."""
    return name.lower().replace(" ", "_")


def _severity_for_result(
    result: LabResult, sentiment_snapshot: dict[str, float]
) -> tuple[str, str | None, float | None]:
    """Return (severity, location_name, sentiment_score) for a result."""
    location = _extract_location_token(result)
    if location:
        slug = _slug_for(location) + "_sentiment"
        score = sentiment_snapshot.get(slug)
        if score is not None:
            return _severity_from_score(score), location, score

    if result.status == "warning":
        return "warning", location, None
    if result.status == "selected" and (result.final_priority_score or 0) >= 0.7:
        return "warning", location, None
    return "info", location, None


def derive_alerts(
    session_id: str,
    scenario: str,
    report: LabRunReport,
    sentiment_snapshot: dict[str, float],
    *,
    now_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Build the alert envelopes for the current refresh.

    `info`-level entries are filtered out — they're not actionable enough to
    surface as alerts. Returns newest entries first (sort by severity, then
    by lab id) so the dashboard can render the most critical at the top.
    """
    ts = now_iso or utcnow_iso()
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    out: list[dict[str, Any]] = []

    for result in list(report.selected) + list(report.warning):
        severity, location, score = _severity_for_result(result, sentiment_snapshot)
        if severity == "info":
            continue
        title = result.lab_name or result.lab_id
        body = result.summary or ""
        recommended_action = None
        if result.recommended_actions:
            first = result.recommended_actions[0]
            ra = first.model_dump() if hasattr(first, "model_dump") else dict(first)
            recommended_action = ra.get("detail") or ra.get("title")

        dedupe_key = f"{session_id}:{result.lab_id}:{(location or 'network').lower()}"

        out.append({
            "alert_id": f"alert-{result.lab_id}-{ts.replace(':', '').replace('-', '')[:14]}",
            "session_id": session_id,
            "scenario": scenario,
            "lab_id": result.lab_id,
            "title": title,
            "body": body,
            "severity": severity,
            "location": location,
            "sentiment_score": score,
            "recommended_action": recommended_action,
            "dedupe_key": dedupe_key,
            "created_at": ts,
        })

    out.sort(key=lambda a: (severity_rank.get(a["severity"], 99), a["lab_id"]))
    return out
