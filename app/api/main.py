"""FastAPI app for the Signal Foundry backend (slice).

Routes:
  GET  /health
  POST /sessions/{session_id}/refresh
  POST /sessions/{session_id}/monitoring-plan/build
  GET  /sessions/{session_id}/monitoring-plan
  GET  /sessions/{session_id}/dashboard
  GET  /sessions/{session_id}/charts/{chart_id}/data
  POST /sessions/{session_id}/bundle/rebuild
  GET  /bundle/{filename}   (static mount, populated by /bundle/rebuild)
  GET  /ui/                  (static dashboard)

Out of scope for this slice: alerts (POST/GET), supply_chain session,
multiple charts, anomaly detection, decision card generation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import synthesize_reply
from app.api.dashboard_spec import CHART_SENTIMENT_TREND, build_dashboard_spec
from app.api.forecasts import build_sales_forecast
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
from app.api.storage import base_dir
from app.export.frontend_bundle import build_bundle
from app.labs.runner import load_demo_context, prepare_context_for_labs

# ---------------------------------------------------------------------------
# Static UI + bundle directories
#
# The bundle lives under the on-disk session-state root so a single
# EXPERIMENT_LAB_API_TMP_DIR override relocates everything for tests.
# `base_dir()` is `tmp/sessions/` in production; placing the bundle under it
# keeps tests hermetic and avoids escaping the env-overridden root.

def _bundle_dir() -> Path:
    return base_dir() / "_bundle"


def _static_ui_dir() -> Path:
    # Ship the static dashboard alongside the package so editable installs
    # and Docker images both find it.
    return Path(__file__).resolve().parent.parent / "static"


app = FastAPI(
    title="Signal Foundry Backend (slice)",
    description=(
        "Vertical slice: session demo_miners / reputation_monitor / "
        "chart sentiment_trend_by_location."
    ),
    version="0.2.0",
)

# CORS: the static dashboard ships under /ui on the same origin, but we also
# accept any localhost origin so the same dashboard can be opened from
# `python -m http.server` on a different port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_ui = _static_ui_dir()
if _static_ui.exists():
    app.mount("/ui", StaticFiles(directory=_static_ui, html=True), name="ui")


def _resolve_session(session_id: str):
    try:
        return get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown session_id: {session_id}") from exc


@app.get("/")
def index() -> RedirectResponse:
    """Convenience redirect to the static dashboard."""
    return RedirectResponse(url="/ui/")


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


@app.get("/sessions/{session_id}/forecasts/sales")
def get_sales_forecast(
    session_id: str,
    horizon_days: int = 30,
    start_date: str | None = None,
) -> dict[str, Any]:
    """Synthetic sales / revenue forecast for the dashboard.

    Returns a deterministic forecast horizon (clamped to 7..90 days) with:
      - per-day baseline / predicted / with-intervention revenue + P10/P90
      - per-day factor decomposition (seasonality, sentiment, competitor, holiday)
      - per-location next-7-day rollup
      - three scenario summaries (do_nothing, add_morning_staff, competitive_promo)
      - model metadata (id, MAPE, MAE, training window, feature list, retrain timing)
      - feature importance + plain-English narrative + anomaly callouts.
    """
    info = _resolve_session(session_id)
    return build_sales_forecast(
        session_id=session_id,
        scenario=info["scenario"],
        horizon_days=horizon_days,
        start_date=start_date,
    )


@app.post("/sessions/{session_id}/chat")
def post_chat(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Stub conversational endpoint used by the dashboard chat widget.

    Not an LLM. The response is synthesized deterministically by routing
    the message through a keyword intent matcher and pulling real numbers
    from the lab snapshot + sales forecast. Always cites the dashboard
    panel its numbers came from.
    """
    info = _resolve_session(session_id)
    message = (body or {}).get("message", "")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="message must be a non-empty string")
    return synthesize_reply(message, session_id=session_id, scenario=info["scenario"])


@app.post("/sessions/{session_id}/bundle/rebuild")
def post_rebuild_bundle(session_id: str) -> dict[str, Any]:
    """Regenerate the static frontend bundle under /bundle/.

    The dashboard calls this after a refresh so its non-live panels
    (findings, labs, predictions, reports) reflect the freshest run.
    """
    _resolve_session(session_id)
    backend_url = os.environ.get("SIGNAL_FOUNDRY_PUBLIC_URL", "")
    written = build_bundle(_bundle_dir(), live_backend_url=backend_url or None)
    return {
        "bundle_dir": str(_bundle_dir()),
        "files": sorted(written.keys()),
    }


# Mount the bundle dir as a static path. It may not exist on first boot;
# create it empty so the mount succeeds.
_bundle = _bundle_dir()
_bundle.mkdir(parents=True, exist_ok=True)
app.mount("/bundle", StaticFiles(directory=_bundle, html=False), name="bundle")
