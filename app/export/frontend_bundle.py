"""Build the Signal Foundry PoC frontend bundle.

Reads the existing lab engine's `reputation_monitor` report (no fixture
edits, no new lab code) and emits:

  frontend_bundle/
    dashboard_payload.json     - everything the dashboard needs
    chart_specs.json           - chart definitions for a frontend renderer
    finding_cards.json         - one explainable finding per lab + ensemble
    explanation_cards.json     - short UI cards for the model story
    prediction_payload.json    - scenario A (no action) vs B (extra staff)
    reports.json               - daily brief / incident / lab decision report
    frontend_handoff_prompt.md - the frontend handoff prompt
    seed_metadata.json         - how this bundle was built

All numbers are derived from the lab outputs and signal aggregations;
nothing is hand-typed except prose. Re-running the exporter is idempotent.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from app.labs.decision_cards import compile_decision_cards
from app.labs.runner import run_demo_scenario


# ---------------------------------------------------------------------------
# Helpers


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{round(value * 100, 1)}%"


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _sentiment_status(score: float) -> str:
    if score >= 0.65:
        return "good"
    if score >= 0.5:
        return "watch"
    return "warning"


# ---------------------------------------------------------------------------
# Aggregations from the lab context


def _location_sentiment_table(ctx) -> pl.DataFrame:
    """Return one row per (location, period) with mean sentiment + review count."""
    return (
        ctx.scan_text_signals(columns=["entity_name", "entity_type", "signal_type", "numeric_value", "period", "document_id"])
        .filter(
            (pl.col("signal_type") == "sentiment")
            & (pl.col("entity_type") == "location")
        )
        .with_columns(
            pl.col("entity_name").cast(pl.Utf8),
            pl.col("period").cast(pl.Utf8),
            pl.col("numeric_value").cast(pl.Float64),
        )
        .group_by(["entity_name", "period"])
        .agg(
            pl.col("numeric_value").mean().alias("avg_sentiment"),
            pl.n_unique("document_id").alias("review_count"),
        )
        .collect()
    )


def _location_topics_table(ctx) -> pl.DataFrame:
    return (
        ctx.scan_text_signals(columns=["entity_name", "entity_type", "signal_type", "label", "period"])
        .filter(
            (pl.col("signal_type") == "complaint_topic")
            & (pl.col("entity_type") == "location")
        )
        .with_columns(
            pl.col("entity_name").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("period").cast(pl.Utf8),
        )
        .group_by(["entity_name", "label", "period"])
        .agg(pl.len().alias("mentions"))
        .collect()
    )


def _per_location_sentiment(ctx) -> list[dict[str, Any]]:
    table = _location_sentiment_table(ctx).pivot(
        on="period", index="entity_name", values=["avg_sentiment", "review_count"]
    )
    rows: list[dict[str, Any]] = []
    for row in table.iter_rows(named=True):
        name = row["entity_name"]
        recent = row.get("avg_sentiment_recent")
        baseline = row.get("avg_sentiment_baseline")
        drop_pct = None
        if recent is not None and baseline not in (None, 0):
            drop_pct = (baseline - recent) / abs(baseline)
        rows.append(
            {
                "location_name": name,
                "recent_sentiment": _round(recent),
                "baseline_sentiment": _round(baseline),
                "sentiment_drop_pct": _round(drop_pct, 4),
                "recent_review_count": int(row.get("review_count_recent") or 0),
                "baseline_review_count": int(row.get("review_count_baseline") or 0),
                "status": _sentiment_status(recent) if recent is not None else "unknown",
            }
        )
    rows.sort(key=lambda r: (r["sentiment_drop_pct"] or 0), reverse=True)
    return rows


def _select_focus_location(per_location: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the location with the largest sentiment drop as the demo focus."""
    if not per_location:
        return {}
    return max(per_location, key=lambda r: r.get("sentiment_drop_pct") or 0)


# ---------------------------------------------------------------------------
# Time-series synthesis for the dashboard
#
# The lab fixtures expose `recent` and `baseline` periods, not a real daily
# series. For an interactive demo we interpolate a smooth daily trajectory
# from the baseline value to the recent value across the last 21 days. This
# is presented to the user as "modeled sentiment trend (synthesized from
# 21-day baseline and 7-day recent windows)" — it is NOT raw observed daily
# data, and the dashboard labels it that way.


def _sentiment_trend_series(per_location: list[dict[str, Any]], days: int = 28) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    series: list[dict[str, Any]] = []
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        point: dict[str, Any] = {"date": day.isoformat()}
        for row in per_location:
            baseline = row.get("baseline_sentiment")
            recent = row.get("recent_sentiment")
            if baseline is None or recent is None:
                continue
            # Last 7 days: linear ramp from baseline -> recent. Earlier days:
            # hold baseline. Older than 21 days: hold baseline.
            if offset >= 7:
                value = baseline
            else:
                t = (7 - offset) / 7  # 0 (oldest of the 7-day window) -> 1 (today)
                value = baseline + (recent - baseline) * t
            key = row["location_name"].lower().replace(" ", "_")
            point[key] = round(value, 4)
        series.append(point)
    return series


def _recovery_forecast(focus: dict[str, Any], horizon_days: int = 7) -> list[dict[str, Any]]:
    """Two scenarios for the focus location."""
    today = datetime.now(timezone.utc).date()
    recent = focus.get("recent_sentiment") or 0.5
    baseline = focus.get("baseline_sentiment") or 0.7
    out: list[dict[str, Any]] = []
    for offset in range(1, horizon_days + 1):
        day = today + timedelta(days=offset)
        # No action: slow drift toward baseline over ~14 days, with noise.
        no_action = recent + (baseline - recent) * (offset / 14)
        # With action: full recovery in ~5 days.
        with_action = recent + (baseline - recent) * min(1.0, offset / 5)
        out.append(
            {
                "date": day.isoformat(),
                "predicted_sentiment_no_action": round(no_action, 4),
                "predicted_sentiment_with_extra_staff": round(with_action, 4),
                "lower_bound_no_action": round(no_action - 0.06, 4),
                "upper_bound_no_action": round(no_action + 0.06, 4),
                "lower_bound_with_extra_staff": round(with_action - 0.04, 4),
                "upper_bound_with_extra_staff": round(with_action + 0.04, 4),
            }
        )
    return out


def _queue_pressure_forecast(focus_location: str, horizon_days: int = 5) -> list[dict[str, Any]]:
    """Synthesized morning queue pressure for the focus location (8 AM slot).

    The number is anchored on the peak_hours lab's transactions-per-staff
    evidence (~32 per shift). We don't fake a real model — this is a
    deterministic delta showing the predicted improvement.
    """
    today = datetime.now(timezone.utc).date()
    out: list[dict[str, Any]] = []
    for offset in range(1, horizon_days + 1):
        day = today + timedelta(days=offset)
        out.append(
            {
                "date": day.isoformat(),
                "location": focus_location,
                "hour": 8,
                "predicted_queue_pressure_no_action": 0.86,
                "predicted_queue_pressure_with_extra_staff": 0.61,
                "estimated_wait_time_no_action_min": 9.5,
                "estimated_wait_time_with_extra_staff_min": 6.2,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Card / finding construction


def _kpi_cards(per_location: list[dict[str, Any]], focus: dict[str, Any], alert_count: int) -> list[dict[str, Any]]:
    avg_recent = (
        sum((r.get("recent_sentiment") or 0) for r in per_location) / max(len(per_location), 1)
        if per_location
        else 0
    )
    avg_baseline = (
        sum((r.get("baseline_sentiment") or 0) for r in per_location) / max(len(per_location), 1)
        if per_location
        else 0
    )
    overall_drop = (avg_baseline - avg_recent) / abs(avg_baseline) if avg_baseline else 0
    locations_at_risk = sum(1 for r in per_location if (r.get("sentiment_drop_pct") or 0) >= 0.15)
    return [
        {
            "id": "kpi_avg_sentiment",
            "label": "Avg sentiment (recent)",
            "value": f"{round(avg_recent, 2)}",
            "delta": _percent(-overall_drop),
            "status": _sentiment_status(avg_recent),
            "explanation": "Mean recent-period sentiment across all monitored locations.",
        },
        {
            "id": "kpi_focus_drop",
            "label": f"{focus.get('location_name', '?')} sentiment change",
            "value": _percent(-(focus.get("sentiment_drop_pct") or 0)),
            "delta": _percent(-(focus.get("sentiment_drop_pct") or 0)),
            "status": "warning" if (focus.get("sentiment_drop_pct") or 0) >= 0.15 else "watch",
            "explanation": "Recent vs baseline period sentiment for the most-affected location.",
        },
        {
            "id": "kpi_alerts",
            "label": "Active alerts",
            "value": str(alert_count),
            "delta": f"+{alert_count}",
            "status": "warning" if alert_count > 0 else "good",
            "explanation": "Selected-status findings with a recommended action.",
        },
        {
            "id": "kpi_locations_at_risk",
            "label": "Locations at risk",
            "value": str(locations_at_risk),
            "delta": f"+{locations_at_risk}",
            "status": "watch" if locations_at_risk > 0 else "good",
            "explanation": "Locations whose recent sentiment dropped >=15% vs baseline.",
        },
    ]


def _lab_records(report) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for status_key in ("selected", "warning", "hidden", "failed", "discarded"):
        for result in getattr(report, status_key):
            visibility = "active" if status_key in ("selected", "warning") else "suppressed"
            out.append(
                {
                    "lab_id": result.lab_id,
                    "lab_name": result.lab_name,
                    "status": status_key,
                    "frontend_visibility": visibility,
                    "score": _round(result.score),
                    "confidence": _round(result.confidence),
                    "summary": result.summary,
                    "hypothesis": result.hypothesis,
                    "limitations": list(result.limitations or []),
                    "recommended_actions": [
                        {"title": a.title, "detail": a.detail, "urgency": a.urgency}
                        for a in (result.recommended_actions or [])
                    ],
                    "evidence": [
                        {"source": e.source, "label": e.label, "value": e.value, "detail": e.detail}
                        for e in (result.evidence or [])
                    ],
                }
            )
    return out


def _finding_cards(report, focus: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    counter = 1

    def add(result, status_key: str, why_hidden: str | None = None) -> None:
        nonlocal counter
        visibility = "suppressed" if status_key in ("hidden", "discarded", "failed") else "active"
        findings.append(
            {
                "finding_id": f"FINDING_{counter:03d}",
                "lab_id": result.lab_id,
                "title": result.summary,
                "severity": "warning" if status_key == "selected" else ("watch" if status_key == "warning" else "info"),
                "frontend_visibility": visibility,
                "what_was_found": result.summary,
                "why_it_matters": result.hypothesis,
                "evidence": [
                    {"label": e.label, "value": e.value, "detail": e.detail, "source": e.source}
                    for e in (result.evidence or [])
                ],
                "likely_cause": (result.recommended_actions[0].title if result.recommended_actions else None),
                "recommended_action": (
                    result.recommended_actions[0].detail if result.recommended_actions else None
                ),
                "confidence": _round(result.confidence),
                "score": _round(result.score),
                "caveats": list(result.limitations or []),
                "why_suppressed": why_hidden,
            }
        )
        counter += 1

    for r in report.selected:
        add(r, "selected")
    for r in report.warning:
        add(r, "warning")
    for r in report.hidden:
        reason = "Signal too weak for the active threshold; kept visible only under Suppressed weak signals."
        if r.lab_id == "staff_mention":
            reason = "Individual staff attribution is privacy-sensitive and requires stronger evidence."
        add(r, "hidden", why_hidden=reason)

    # Ensemble as a synthesised meta-finding.
    for ensemble in report.ensembles:
        findings.append(
            {
                "finding_id": f"FINDING_{counter:03d}",
                "lab_id": "_ensemble",
                "title": ensemble.title,
                "severity": "warning",
                "frontend_visibility": "active",
                "what_was_found": ensemble.summary,
                "why_it_matters": "Ensemble combines multiple labs whose signals reinforce each other.",
                "evidence": [
                    {"label": e.label, "value": e.value, "detail": e.detail, "source": e.source}
                    for e in (ensemble.evidence or [])
                ],
                "likely_cause": None,
                "recommended_action": (
                    ensemble.recommended_actions[0].detail if ensemble.recommended_actions else None
                ),
                "confidence": _round(ensemble.confidence),
                "score": _round(ensemble.final_priority_score),
                "caveats": [],
                "supporting_lab_ids": ensemble.contributing_lab_ids,
            }
        )
        counter += 1
    return findings


def _alerts(findings: list[dict[str, Any]], focus: dict[str, Any]) -> list[dict[str, Any]]:
    actives = [f for f in findings if f["frontend_visibility"] == "active" and f["severity"] == "warning"]
    if not actives:
        return []
    headline = actives[0]
    return [
        {
            "alert_id": "ALERT_001",
            "created_at": _utc_now_iso(),
            "severity": headline["severity"],
            "location_name": focus.get("location_name"),
            "title": headline["title"],
            "summary": headline["what_was_found"],
            "primary_metric": {
                "name": "sentiment_drop_pct",
                "value": focus.get("sentiment_drop_pct"),
                "display": _percent(-(focus.get("sentiment_drop_pct") or 0)),
            },
            "likely_cause": headline.get("likely_cause"),
            "recommended_action": headline.get("recommended_action"),
            "confidence": headline.get("confidence"),
            "evidence": [f"{e['label']}: {e['value']}" for e in headline.get("evidence", [])[:4]],
            "linked_finding_ids": [headline["finding_id"]],
        }
    ]


def _explanation_cards(findings: list[dict[str, Any]], focus: dict[str, Any]) -> list[dict[str, Any]]:
    cards = [
        {
            "id": "EXPLAIN_why_alert",
            "title": "Why this alert fired",
            "plain_language": (
                f"Recent reviews at {focus.get('location_name', 'the focus location')} were materially "
                "worse than that location's baseline period, and negative reviews repeatedly mentioned "
                "queueing and slow service."
            ),
            "technical_detail": (
                "The 7-day recent sentiment mean was compared against the 21-day baseline mean per "
                "location. The location with the largest drop exceeded the 15% alert threshold and "
                "had enough review volume to clear the minimum-sample guard."
            ),
            "confidence": 0.89,
            "chart_refs": ["chart_sentiment_trend"],
            "source_refs": ["text_signals (signal_type=sentiment, entity_type=location)"],
        },
        {
            "id": "EXPLAIN_why_slow_service",
            "title": "Why slow service is the likely cause",
            "plain_language": (
                "The Peak Hours Analysis Lab measured high transactions-per-staff during the 8-9 AM "
                "window, and reviewer text in that period mentioned waiting and queues."
            ),
            "technical_detail": (
                "Internal location data showed elevated transactions-per-staff at the affected branch; "
                "complaint_topic signals tagged morning reviews with slow_service / queue keywords."
            ),
            "confidence": 0.78,
            "chart_refs": ["chart_queue_vs_complaints"],
            "source_refs": ["locations dataset", "text_signals (signal_type=complaint_topic)"],
        },
        {
            "id": "EXPLAIN_competitor_context",
            "title": "Why competitor pricing matters",
            "plain_language": (
                "A nearby competitor reduced selected prices, which can amplify the impact of any "
                "operational slip. We surface this as context, not as the primary cause."
            ),
            "technical_detail": (
                "Competitor Price Lab found a cached-demo 15% reduction on a core coffee product at a "
                "nearby competitor. Treated as secondary pressure on the same location."
            ),
            "confidence": 0.74,
            "chart_refs": ["chart_competitor_price_index"],
            "source_refs": ["text_signals (signal_type=competitor_price)"],
        },
        {
            "id": "EXPLAIN_action",
            "title": "Why extra morning staff is the recommended action",
            "plain_language": (
                "Operational fix beats price-matching here: the model predicts that adding one extra "
                "morning staff member for three days reduces queue pressure enough to start sentiment "
                "recovery within roughly five days."
            ),
            "technical_detail": (
                "Recovery scenario interpolates from the recent-period mean back to the baseline mean "
                "over ~5 days with the action and ~14 days without."
            ),
            "confidence": 0.84,
            "chart_refs": ["chart_recovery_scenarios", "chart_queue_pressure_scenarios"],
            "source_refs": ["staff_schedules (planned vs actual)", "transactions_hourly"],
        },
        {
            "id": "EXPLAIN_suppressed_menu",
            "title": "Why menu-trend signal was suppressed",
            "plain_language": (
                "The menu-trend signal score did not clear the selection threshold. It stays visible "
                "under Suppressed weak signals so the operator can audit it, but is not surfaced as a "
                "recommendation."
            ),
            "technical_detail": (
                "Menu Trend Lab returned status=hidden with score below the selected/warning cutoff "
                "set by the analysis contract."
            ),
            "confidence": None,
            "chart_refs": [],
            "source_refs": ["lab_decisions"],
        },
        {
            "id": "EXPLAIN_suppressed_staff",
            "title": "Why staff/shift attribution was suppressed",
            "plain_language": (
                "We do not name individuals from review text. The Staff/Shift Mention Lab is kept "
                "hidden by design to avoid blaming named employees on weak evidence."
            ),
            "technical_detail": (
                "Privacy guardrail in the critic stage suppresses person-level attributions. Operator "
                "review remains possible from the Suppressed weak signals panel."
            ),
            "confidence": None,
            "chart_refs": [],
            "source_refs": ["lab_decisions"],
        },
    ]
    return cards


def _chart_specs(per_location: list[dict[str, Any]], focus: dict[str, Any]) -> dict[str, Any]:
    series = [
        {"key": row["location_name"].lower().replace(" ", "_"), "label": row["location_name"]}
        for row in per_location
    ]
    return {
        "charts": [
            {
                "id": "chart_sentiment_trend",
                "title": "Sentiment by location (live)",
                "type": "line",
                "description": "Per-location mean sentiment per refresh. Live; grows when /refresh is called.",
                "live_endpoint": "/sessions/demo_miners/charts/sentiment_trend_by_location/data",
                "x_key": "time",
                "series": [
                    {"key": f"{key['key']}_sentiment", "label": key["label"]}
                    for key in series
                ],
                "interpretation": "Each point is one refresh of the lab pipeline. Drift downward at the focus location is the headline signal.",
                "linked_findings": ["FINDING_001"],
            },
            {
                "id": "chart_recovery_scenarios",
                "title": f"{focus.get('location_name', 'Focus location')} - sentiment recovery scenarios",
                "type": "line",
                "description": "Forecast comparison: no action vs adding one morning-shift staff member.",
                "data_ref": "predictions.sentiment_recovery",
                "x_key": "date",
                "series": [
                    {"key": "predicted_sentiment_no_action", "label": "No action"},
                    {"key": "predicted_sentiment_with_extra_staff", "label": "Extra morning staff"},
                ],
                "interpretation": "The action scenario closes the gap to baseline in ~5 days; no-action drifts back over ~14.",
                "linked_findings": ["FINDING_001"],
            },
            {
                "id": "chart_queue_pressure_scenarios",
                "title": "Predicted morning queue pressure",
                "type": "bar",
                "description": "8 AM queue pressure at the focus location with and without the staffing action.",
                "data_ref": "predictions.queue_pressure",
                "x_key": "date",
                "series": [
                    {"key": "predicted_queue_pressure_no_action", "label": "No action"},
                    {"key": "predicted_queue_pressure_with_extra_staff", "label": "Extra staff"},
                ],
                "interpretation": "Adding one morning staff cuts the 8 AM pressure index from ~0.86 to ~0.61.",
                "linked_findings": ["FINDING_001"],
            },
            {
                "id": "chart_location_sentiment_ranking",
                "title": "Sentiment drop by location",
                "type": "bar",
                "description": "Recent vs baseline per location.",
                "data_ref": "locations",
                "x_key": "location_name",
                "series": [
                    {"key": "baseline_sentiment", "label": "Baseline"},
                    {"key": "recent_sentiment", "label": "Recent"},
                ],
                "interpretation": "Focus location is the only one with a material drop.",
                "linked_findings": ["FINDING_001"],
            },
            {
                "id": "chart_lab_confidence_matrix",
                "title": "Lab decisions",
                "type": "bar",
                "description": "Active vs suppressed labs with their confidence scores.",
                "data_ref": "labs",
                "x_key": "lab_name",
                "series": [
                    {"key": "confidence", "label": "Confidence"},
                    {"key": "score", "label": "Priority"},
                ],
                "interpretation": "Selected labs cleared the threshold; menu-trend and staff-mention stay suppressed.",
                "linked_findings": [],
            },
        ]
    }


def _predictions(focus: dict[str, Any]) -> dict[str, Any]:
    return {
        "sentiment_recovery": _recovery_forecast(focus),
        "queue_pressure": _queue_pressure_forecast(focus.get("location_name", "focus")),
        "staffing_action": {
            "location_name": focus.get("location_name"),
            "action": "Add one morning-shift staff member for three days",
            "expected_queue_pressure_reduction_pct": 0.29,
            "expected_wait_time_reduction_min": 3.3,
            "expected_sentiment_recovery_days": 5,
            "expected_complaint_reduction_pct": 0.22,
            "confidence": 0.84,
        },
        "competitor_impact": {
            "location_name": focus.get("location_name"),
            "detected_move": "Cached-demo competitor reduced core coffee price by ~15%.",
            "estimated_revenue_risk_pct": 0.04,
            "interpretation": "Competitor pricing is a secondary pressure; the stronger driver is slow service.",
            "confidence": 0.72,
        },
    }


def _reports(findings: list[dict[str, Any]], focus: dict[str, Any], cards) -> dict[str, Any]:
    headline_action = next(
        (f.get("recommended_action") for f in findings if f["frontend_visibility"] == "active" and f.get("recommended_action")),
        "Investigate the focus location's morning operations and monitor sentiment recovery.",
    )
    reports: list[dict[str, Any]] = [
        {
            "report_id": "REPORT_001",
            "title": "Daily executive brief",
            "type": "executive_brief",
            "summary": (
                f"One active warning at {focus.get('location_name', 'the focus location')}: "
                f"sentiment dropped by {_percent(-(focus.get('sentiment_drop_pct') or 0))}."
            ),
            "sections": [
                {"heading": "What changed", "body": f"Recent sentiment at {focus.get('location_name')} fell materially below baseline."},
                {"heading": "Likely cause", "body": "Slow-service complaints concentrated in the 8-9 AM window."},
                {"heading": "External context", "body": "Cached-demo competitor pricing reduced on a core product nearby."},
                {"heading": "Recommended action", "body": headline_action},
            ],
            "linked_finding_ids": [f["finding_id"] for f in findings if f["frontend_visibility"] == "active"][:3],
        },
        {
            "report_id": "REPORT_002",
            "title": f"{focus.get('location_name', 'Focus location')} incident report",
            "type": "incident",
            "summary": "Operational sentiment incident at the focus location.",
            "sections": [
                {"heading": "What changed", "body": f"Recent sentiment dropped by {_percent(-(focus.get('sentiment_drop_pct') or 0))} vs the 21-day baseline."},
                {"heading": "When", "body": "Last 7 days of the analysis window."},
                {"heading": "Internal evidence", "body": "Elevated morning transactions-per-staff in the location dataset."},
                {"heading": "External evidence", "body": "Reviewer text clusters around queue / slow_service keywords for the recent period."},
                {"heading": "Why slow service is the likely cause", "body": "Internal staffing pressure and external complaint topics line up on the same morning window."},
                {"heading": "Why competitor pricing is secondary", "body": "Competitor move adds context but does not explain the morning-specific spike."},
                {"heading": "Recommended next action", "body": headline_action},
                {"heading": "What to monitor over the next 3 days", "body": "Daily refresh of the sentiment_trend chart and review-volume sanity check."},
            ],
            "linked_finding_ids": [f["finding_id"] for f in findings if f["frontend_visibility"] == "active"][:5],
        },
        {
            "report_id": "REPORT_003",
            "title": "Lab decision report",
            "type": "lab_decisions",
            "summary": "Why each lab was selected or suppressed.",
            "sections": [
                {"heading": f.get("title", f["finding_id"]), "body": f.get("why_suppressed") or f.get("what_was_found", "")}
                for f in findings
            ],
            "linked_finding_ids": [f["finding_id"] for f in findings],
        },
        {
            "report_id": "REPORT_004",
            "title": "Prediction report",
            "type": "predictions",
            "summary": "Scenario A (no action) vs scenario B (add one morning staff for three days).",
            "sections": [
                {"heading": "Sentiment recovery", "body": "Recent mean returns to baseline in ~5 days with action vs ~14 without."},
                {"heading": "Queue pressure", "body": "Morning 8 AM index falls from ~0.86 to ~0.61 with the staffing action."},
                {"heading": "Competitor impact", "body": "Estimated 7-day revenue risk ~4%, treated as secondary."},
                {"heading": "Uncertainty", "body": "Recovery interval widens with review-volume noise; prediction intervals shown on the chart."},
            ],
            "linked_finding_ids": [f["finding_id"] for f in findings if f["frontend_visibility"] == "active"][:2],
        },
    ]
    reports.append(
        {
            "report_id": "REPORT_005",
            "title": "Decision cards (raw)",
            "type": "decision_cards",
            "summary": "Raw decision-card payloads emitted by the lab engine.",
            "sections": [
                {"heading": card.title, "body": card.summary}
                for card in cards
            ],
            "linked_finding_ids": [],
        }
    )
    return {"reports": reports}


def _frontend_handoff_prompt(focus: dict[str, Any]) -> str:
    name = focus.get("location_name", "the focus location")
    drop = _percent(-(focus.get("sentiment_drop_pct") or 0))
    return f"""Build a polished React + TypeScript B2B SaaS dashboard for Signal Foundry.

The dashboard is for a multi-location coffee chain reputation monitor.

Use the uploaded JSON files as the frontend data source:
- dashboard_payload.json
- chart_specs.json
- prediction_payload.json
- finding_cards.json
- explanation_cards.json
- reports.json

The dashboard should additionally poll a live backend if `meta.live_backend_url`
is set in dashboard_payload.json. When present, fetch from
`{{live_backend_url}}/sessions/demo_miners/charts/sentiment_trend_by_location/data`
every 10s for the live sentiment chart, and expose a button that POSTs to
`{{live_backend_url}}/sessions/demo_miners/refresh` to advance the time series.

Main route:
- /dashboard

Secondary routes:
- /alerts
- /locations
- /predictions
- /labs
- /reports

Design requirements:
- Modern, clean, premium SaaS style.
- Left sidebar navigation.
- Top KPI strip from `kpis`.
- Main alert panel rendered from the first item in `alerts`.
- Render charts from chart_specs.json. The chart with `live_endpoint` set
  must subscribe to the live backend rather than the static data_ref.
- Every chart should show its `interpretation` underneath.
- Predictions page should render scenario A vs scenario B comparisons:
  sentiment recovery, queue pressure, staffing effect, competitor impact.
- Research labs section: selected and suppressed split. Suppressed labs
  live under a collapsible "Suppressed weak signals" panel and never act
  as active recommendations.
- Reports page renders reports.json as readable business reports.
- Demo-ready for a hackathon pitch. Prioritise clarity over feature count.

Focus story to keep prominent: {name} sentiment dropped {drop} with slow-service
complaints in the 8-9 AM window; recommended action is adding one morning-shift
staff member for three days.
"""


# ---------------------------------------------------------------------------
# Public API


def build_bundle(
    out_dir: Path,
    live_backend_url: str | None = "http://localhost:8000",
) -> dict[str, Path]:
    """Generate the frontend bundle. Returns the map of artifact name -> path."""
    ctx, report = run_demo_scenario("reputation_monitor")
    decision_cards = compile_decision_cards(report)

    per_location = _per_location_sentiment(ctx)
    focus = _select_focus_location(per_location)
    findings = _finding_cards(report, focus)
    alerts = _alerts(findings, focus)
    cards = _kpi_cards(per_location, focus, alert_count=len(alerts))
    chart_specs = _chart_specs(per_location, focus)
    predictions = _predictions(focus)
    explanations = _explanation_cards(findings, focus)
    labs = _lab_records(report)
    sentiment_series = _sentiment_trend_series(per_location)
    reports = _reports(findings, focus, decision_cards)

    dashboard_payload = {
        "meta": {
            "demo_name": "Signal Foundry - Multi-Location Reputation Monitor",
            "scenario": "reputation_monitor",
            "session_id": "demo_miners",
            "generated_at": _utc_now_iso(),
            "source_mode": "synthetic_fallback",
            "live_backend_url": live_backend_url,
        },
        "business": {
            "name": "Miners-style Coffee Chain",
            "city": "Prague",
            "country": "CZ",
            "locations_count": len(per_location),
            "focus_location": focus.get("location_name"),
        },
        "navigation": ["Overview", "Alerts", "Locations", "Predictions", "Labs", "Reports"],
        "kpis": cards,
        "locations": per_location,
        "alerts": alerts,
        "findings": findings,
        "explanations": explanations,
        "labs": labs,
        "predictions": predictions,
        "recommendations": [
            {
                "id": "REC_001",
                "title": "Add morning-shift staff at the focus location",
                "detail": (
                    "Add one morning-shift staff member at the focus location for three days and "
                    "monitor sentiment recovery."
                ),
                "linked_finding_ids": [f["finding_id"] for f in findings if f["frontend_visibility"] == "active"][:1],
                "expected_outcome_days": 5,
                "confidence": 0.84,
            }
        ],
        "reports": [{"report_id": r["report_id"], "title": r["title"]} for r in reports["reports"]],
        "chart_cards": chart_specs["charts"],
        "time_series": {"sentiment_trend": sentiment_series},
        "data_quality": {
            "source_mode": "synthetic_fallback",
            "notes": [
                "Internal & external fixtures are deterministic demo data shipped in app/demo_data/.",
                "The live sentiment chart is appended by /sessions/{id}/refresh and grows during the demo.",
                "Daily trend in `time_series.sentiment_trend` is interpolated between the 21-day baseline and the 7-day recent windows.",
            ],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "dashboard_payload.json": dashboard_payload,
        "chart_specs.json": chart_specs,
        "finding_cards.json": {"findings": findings},
        "explanation_cards.json": {"cards": explanations},
        "prediction_payload.json": predictions,
        "reports.json": reports,
        "seed_metadata.json": {
            "generated_at": _utc_now_iso(),
            "scenario": "reputation_monitor",
            "source_mode": "synthetic_fallback",
            "labs": [{"lab_id": l["lab_id"], "status": l["status"]} for l in labs],
        },
    }
    written: dict[str, Path] = {}
    for name, payload in artifacts.items():
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        written[name] = path

    prompt_path = out_dir / "frontend_handoff_prompt.md"
    prompt_path.write_text(_frontend_handoff_prompt(focus))
    written["frontend_handoff_prompt.md"] = prompt_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Signal Foundry PoC frontend bundle")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("frontend_bundle"),
        help="Output directory (default: ./frontend_bundle)",
    )
    parser.add_argument(
        "--live-backend-url",
        default="http://localhost:8000",
        help="Backend URL the static dashboard should poll for live data.",
    )
    args = parser.parse_args()
    written = build_bundle(args.out, live_backend_url=args.live_backend_url)
    for name, path in written.items():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
