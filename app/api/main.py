"""FastAPI app for the Signal Foundry backend (slice).

Routes:
  GET  /health
  POST /sessions/{session_id}/refresh
  POST /sessions/{session_id}/monitoring-plan/build
  GET  /sessions/{session_id}/monitoring-plan
  GET  /sessions/{session_id}/dashboard
  GET  /sessions/{session_id}/charts/{chart_id}/data

Out of scope for this slice: alerts (POST/GET), supply_chain session,
multiple charts, anomaly detection, decision card generation.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from app.api.dashboard_spec import CHART_SENTIMENT_TREND, build_dashboard_spec
from app.api.history_store import read_metric_series
from app.api.monitoring_plan import load_monitoring_plan
from app.api.refresh import build_only, refresh_session
from app.api.schemas import (
    ChartDataResponse,
    DashboardSpec,
    MonitoringPlan,
    MonitoringPlanBuildRequest,
    RefreshRequest,
    RefreshResponse,
)
from app.api.sentiment_metrics import location_sentiment_snapshot
from app.api.sessions import get_session
from app.labs.runner import load_demo_context, prepare_context_for_labs

app = FastAPI(
    title="Signal Foundry Backend (slice)",
    description=(
        "Vertical slice: session demo_miners / reputation_monitor / "
        "chart sentiment_trend_by_location."
    ),
    version="0.1.0",
)


def _resolve_session(session_id: str):
    try:
        return get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown session_id: {session_id}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions/{session_id}/refresh", response_model=RefreshResponse)
def post_refresh(session_id: str, body: RefreshRequest | None = None) -> RefreshResponse:
    info = _resolve_session(session_id)
    scenario = (body.scenario if body else None) or info["scenario"]
    response, _ = refresh_session(session_id, scenario)
    return response


@app.post("/sessions/{session_id}/monitoring-plan/build", response_model=MonitoringPlan)
def post_build_plan(
    session_id: str, body: MonitoringPlanBuildRequest | None = None
) -> MonitoringPlan:
    info = _resolve_session(session_id)
    scenario = (body.scenario if body else None) or info["scenario"]
    return build_only(session_id, scenario)


@app.get("/sessions/{session_id}/monitoring-plan", response_model=MonitoringPlan)
def get_monitoring_plan(session_id: str) -> MonitoringPlan:
    _resolve_session(session_id)
    plan = load_monitoring_plan(session_id)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"no monitoring plan persisted for {session_id}; call /monitoring-plan/build or /refresh first",
        )
    return plan


@app.get("/sessions/{session_id}/dashboard", response_model=DashboardSpec)
def get_dashboard(session_id: str) -> DashboardSpec:
    info = _resolve_session(session_id)
    plan = load_monitoring_plan(session_id)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"no monitoring plan persisted for {session_id}; call /monitoring-plan/build or /refresh first",
        )
    context = prepare_context_for_labs(load_demo_context(info["scenario"]))
    snapshot = location_sentiment_snapshot(context)
    series_points = read_metric_series(session_id, CHART_SENTIMENT_TREND, limit=1)
    last_updated = series_points[-1]["time"] if series_points else ""
    return build_dashboard_spec(
        session_id=session_id,
        scenario=info["scenario"],
        title=info["title"],
        plan=plan,
        sentiment_snapshot=snapshot,
        last_updated=last_updated,
        headline=None,
    )


@app.get("/sessions/{session_id}/charts/{chart_id}/data", response_model=ChartDataResponse)
def get_chart_data(session_id: str, chart_id: str, limit: int = 200) -> ChartDataResponse:
    _resolve_session(session_id)
    if chart_id != CHART_SENTIMENT_TREND:
        raise HTTPException(status_code=404, detail=f"unknown chart_id: {chart_id}")
    points: list[dict[str, Any]] = read_metric_series(session_id, chart_id, limit=limit)
    return ChartDataResponse(chart_id=chart_id, data=points)
