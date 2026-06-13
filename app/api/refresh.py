"""Run the lab pipeline for a session, append a point to chart history, and
return a refresh summary.

This is the single entry point the FastAPI `/refresh` route calls and that
the `/dashboard` builder relies on for the latest snapshot.
"""
from __future__ import annotations

from typing import Any

from app.api.artifacts import build_prediction_artifacts
from app.api.dashboard_spec import CHART_SENTIMENT_TREND
from app.api.history_store import append_metric_point, utcnow_iso
from app.api.monitoring_plan import (
    build_monitoring_plan,
    load_monitoring_plan,
    save_monitoring_plan,
)
from app.api.schemas import (
    AlertEnvelope,
    DashboardDelta,
    MonitoringPlan,
    RefreshResponse,
)
from app.api.sentiment_metrics import location_sentiment_snapshot
from app.labs.runner import run_demo_scenario


def _model_record(result) -> dict[str, Any]:
    return {
        "lab_id": result.lab_id,
        "lab_name": result.lab_name,
        "status": result.status,
        "score": result.score,
        "confidence": result.confidence,
        "summary": result.summary,
    }


def refresh_session(session_id: str, scenario: str) -> tuple[RefreshResponse, MonitoringPlan]:
    """Run labs, refresh dashboard chart data, persist artifacts.

    Returns the response payload plus the (newly-saved) monitoring plan so
    the caller can decide what to render. The plan is rebuilt from every
    refresh — there is no separate "build vs refresh" distinction yet.
    """
    context, report = run_demo_scenario(scenario)
    artifacts = build_prediction_artifacts(report)
    plan = build_monitoring_plan(session_id, scenario, artifacts)
    save_monitoring_plan(plan)

    timestamp = utcnow_iso()
    snapshot = location_sentiment_snapshot(context)

    updated_chart_ids: list[str] = []
    if snapshot:
        point: dict[str, Any] = {"time": timestamp, **snapshot}
        append_metric_point(session_id, CHART_SENTIMENT_TREND, point)
        updated_chart_ids.append(CHART_SENTIMENT_TREND)

    response = RefreshResponse(
        session_id=session_id,
        scenario=scenario,
        prediction_changed=bool(artifacts),
        models=[_model_record(r) for r in report.selected + report.warning],
        anomalies=[],
        alert=AlertEnvelope(should_notify=False),
        decision_cards=[],
        dashboard_delta=DashboardDelta(
            updated_chart_ids=updated_chart_ids,
            last_updated=timestamp,
        ),
    )
    return response, plan


def build_only(session_id: str, scenario: str) -> MonitoringPlan:
    """Run labs and persist a monitoring plan without touching chart history.

    Used by `/monitoring-plan/build` when the caller wants the plan refreshed
    but does not want to advance the dashboard time series.
    """
    existing = load_monitoring_plan(session_id)
    if existing is not None and existing.scenario == scenario:
        # No-op rebuild: we still re-run labs so the plan reflects current
        # selection. This keeps the build endpoint idempotent in shape but
        # not in side effects.
        pass
    _, report = run_demo_scenario(scenario)
    artifacts = build_prediction_artifacts(report)
    plan = build_monitoring_plan(session_id, scenario, artifacts)
    save_monitoring_plan(plan)
    return plan
