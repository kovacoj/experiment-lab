"""Build, persist, and load MonitoringPlans for a session."""
from __future__ import annotations

from app.api.schemas import MonitoredModelSpec, MonitoringPlan, PredictionArtifact
from app.api.storage import monitoring_plan_path, read_json, write_json

# Map model_id → chart_ids that depend on it. Add new entries as charts are
# introduced; do not synthesize chart bindings from artifact metadata.
_MODEL_CHART_BINDINGS: dict[str, list[str]] = {
    "location_sentiment_drop_v1": ["sentiment_trend_by_location"],
}

_MODEL_ALERT_POLICIES: dict[str, str] = {
    "location_sentiment_drop_v1": "reputation_sentiment_drop_policy",
}


def build_monitoring_plan(
    session_id: str,
    scenario: str,
    artifacts: list[PredictionArtifact],
) -> MonitoringPlan:
    monitored: list[MonitoredModelSpec] = []
    for artifact in artifacts:
        if not artifact.monitoring_eligible:
            continue
        if artifact.status not in ("selected", "warning"):
            continue
        priority = "primary" if artifact.status == "selected" else "secondary"
        monitored.append(
            MonitoredModelSpec(
                model_id=artifact.model_id,
                lab_id=artifact.lab_id,
                scenario=artifact.scenario,
                prediction_name=artifact.prediction_name,
                entity_scope=artifact.entity_scope,
                refresh_minutes=artifact.recommended_refresh_minutes,
                chart_ids=_MODEL_CHART_BINDINGS.get(artifact.model_id, []),
                alert_policy_id=_MODEL_ALERT_POLICIES.get(artifact.model_id),
                priority=priority,
                metadata=artifact.metadata,
            )
        )

    return MonitoringPlan(
        monitoring_plan_id=f"{session_id}:{scenario}:v1",
        session_id=session_id,
        scenario=scenario,
        monitored_models=monitored,
        refresh_endpoint=f"/sessions/{session_id}/refresh",
        dashboard_endpoint=f"/sessions/{session_id}/dashboard",
        alert_endpoint=f"/sessions/{session_id}/alerts",
    )


def save_monitoring_plan(plan: MonitoringPlan) -> None:
    write_json(monitoring_plan_path(plan.session_id), plan.model_dump())


def load_monitoring_plan(session_id: str) -> MonitoringPlan | None:
    payload = read_json(monitoring_plan_path(session_id))
    if payload is None:
        return None
    return MonitoringPlan.model_validate(payload)
