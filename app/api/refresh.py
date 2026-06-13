"""Run the lab pipeline for a session, append a point to chart history, and
return a refresh summary.

This is the single entry point the FastAPI `/refresh` route calls and that
the `/dashboard` builder relies on for the latest snapshot.
"""
from __future__ import annotations

from typing import Any

from app.api import user_context as user_context_store
from app.api.alerts import derive_alerts
from app.api.apify_client import fetch_dataset_items, run_actor
from app.api.apify_ingestion import normalize_apify_reviews
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
from app.api.storage import alerts_log_path, decision_cards_path, read_json, write_json
from app.labs.decision_cards import compile_decision_cards
from app.labs.runner import (
    build_report,
    load_demo_context,
    prepare_context_for_labs,
    run_all_labs,
    run_demo_scenario,
)
from app.labs.schemas import DataSource
from app.text_engine.source_adapters import load_raw_external_records


def _model_record(result) -> dict[str, Any]:
    return {
        "lab_id": result.lab_id,
        "lab_name": result.lab_name,
        "status": result.status,
        "score": result.score,
        "confidence": result.confidence,
        "summary": result.summary,
    }


def _external_stream_config(streams: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the ``streams.external`` block if it requests Apify ingestion."""
    if not streams:
        return None
    external = streams.get("external")
    if not isinstance(external, dict):
        return None
    if not (external.get("apify_dataset_id") or external.get("apify_actor_id")):
        return None
    return external


def _build_apify_augmented_context(scenario: str, external: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Pull items from Apify, merge into demo external records, return context + provenance.

    The lab context's ``external_data`` flips from a file path to an
    in-memory ``DataSource(records=...)`` so the pipeline doesn't need to
    touch disk for the merged stream. The on-disk demo fixture is **read
    once** (never overwritten) and combined with normalized Apify rows.
    """
    base = load_demo_context(scenario)
    demo_records = load_raw_external_records(base.external_data.path)  # type: ignore[arg-type]

    max_items = int(external.get("max_items", 20))
    dataset_id = external.get("apify_dataset_id")
    actor_id = external.get("apify_actor_id")

    if dataset_id:
        apify_result = fetch_dataset_items(str(dataset_id), max_items=max_items)
    else:
        apify_result = run_actor(str(actor_id), max_items=max_items)

    normalized = normalize_apify_reviews(apify_result.get("items", []))
    merged_records: list[dict[str, object]] = list(demo_records) + list(normalized)

    augmented = base.model_copy(
        update={
            "external_data": DataSource(records=merged_records),
            "metadata": {
                **base.metadata,
                "apify_mode": apify_result.get("mode"),
                "apify_actor_id": apify_result.get("actor_id"),
                "apify_actor_run_id": apify_result.get("actor_run_id"),
                "apify_items_received": len(apify_result.get("items", [])),
                "apify_items_ingested": len(normalized),
                "apify_error": apify_result.get("error"),
            },
        }
    )
    provenance = {
        "mode": apify_result.get("mode"),
        "actor_id": apify_result.get("actor_id"),
        "actor_run_id": apify_result.get("actor_run_id"),
        "received": len(apify_result.get("items", [])),
        "ingested": len(normalized),
        "error": apify_result.get("error"),
    }
    return augmented, provenance


def refresh_session(
    session_id: str,
    scenario: str,
    *,
    streams: dict[str, Any] | None = None,
) -> tuple[RefreshResponse, MonitoringPlan]:
    """Run labs, refresh dashboard chart data, persist artifacts.

    Returns the response payload plus the (newly-saved) monitoring plan so
    the caller can decide what to render. The plan is rebuilt from every
    refresh — there is no separate "build vs refresh" distinction yet.

    If ``streams.external`` requests Apify ingestion the demo external
    fixture is merged with normalized Apify rows before the labs run.
    """
    external_cfg = _external_stream_config(streams)
    if external_cfg is None:
        context, report = run_demo_scenario(scenario)
        apify_provenance: dict[str, Any] | None = None
    else:
        augmented, apify_provenance = _build_apify_augmented_context(scenario, external_cfg)
        results = run_all_labs(scenario, augmented)
        report = build_report(augmented, results)
        # `run_demo_scenario` returns the prepared context (with
        # text_documents/text_signals populated) so the dashboard snapshot
        # can read them. Mirror that here for the Apify branch.
        context = prepare_context_for_labs(augmented)
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

    # Compile + persist decision cards so GET /sessions/{id}/cards is cheap
    # and doesn't re-run the labs.
    cards = compile_decision_cards(report)
    cards_payload = [c.model_dump() for c in cards]
    write_json(decision_cards_path(session_id), {
        "session_id": session_id,
        "scenario": scenario,
        "generated_at": timestamp,
        "cards": cards_payload,
    })

    # Derive + append alerts for this refresh
    new_alerts = derive_alerts(session_id, scenario, report, snapshot, now_iso=timestamp)
    if new_alerts:
        existing = read_json(alerts_log_path(session_id)) or {"alerts": []}
        history = list(existing.get("alerts", []))
        history.extend(new_alerts)
        # Cap to last 200 alert entries to keep the file bounded.
        write_json(alerts_log_path(session_id), {"alerts": history[-200:]})

    # Headline alert envelope: most critical alert wins
    top_alert = new_alerts[0] if new_alerts else None
    alert_env = AlertEnvelope(
        should_notify=bool(top_alert),
        title=top_alert.get("title") if top_alert else None,
        body=top_alert.get("body") if top_alert else None,
        severity=top_alert.get("severity") if top_alert else None,
        recommended_action=top_alert.get("recommended_action") if top_alert else None,
        dedupe_key=top_alert.get("dedupe_key") if top_alert else None,
    )

    response = RefreshResponse(
        session_id=session_id,
        scenario=scenario,
        prediction_changed=bool(artifacts),
        models=[_model_record(r) for r in report.selected + report.warning],
        anomalies=[],
        alert=alert_env,
        decision_cards=cards_payload,
        dashboard_delta=DashboardDelta(
            updated_chart_ids=updated_chart_ids,
            last_updated=timestamp,
        ),
        external_provenance=apify_provenance,
        user_context=user_context_store.recent_context_summary(session_id),
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
