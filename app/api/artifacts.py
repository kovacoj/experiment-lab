"""Derive PredictionArtifact records from a LabRunReport.

Slice scope: only the `location_sentiment` lab produces an artifact. Other
labs run (they have to, to populate the report) but do not yet have artifact
mappings. Adding a mapping for a new lab is a one-function change.
"""
from __future__ import annotations

from collections.abc import Callable

from app.api.schemas import ArtifactStatus, PredictionArtifact
from app.labs.schemas import LabResult, LabRunReport

ArtifactBuilder = Callable[[LabResult, ArtifactStatus], list[PredictionArtifact]]


def _build_location_sentiment(result: LabResult, status: ArtifactStatus) -> list[PredictionArtifact]:
    monitoring_rules = result.monitoring_rules or []
    entity = None
    for rule in monitoring_rules:
        if rule.get("type") == "sentiment_drop":
            entity = rule.get("entity_name")
            break
    return [
        PredictionArtifact(
            model_id="location_sentiment_drop_v1",
            lab_id=result.lab_id,
            scenario=result.scenario,
            prediction_name="sentiment_drop_probability",
            entity_scope="location",
            metric_keys=["sentiment_score", "negative_sentiment_share"],
            output_metric="sentiment_score",
            current_value=result.score,
            confidence=result.confidence,
            status=status,
            monitoring_eligible=True,
            chart_eligible=True,
            alert_eligible=True,
            recommended_refresh_minutes=60,
            metadata={"focus_entity": entity} if entity else {},
        )
    ]


_BUILDERS: dict[str, ArtifactBuilder] = {
    "location_sentiment": _build_location_sentiment,
}


def build_prediction_artifacts(report: LabRunReport) -> list[PredictionArtifact]:
    """Convert selected/warning lab results into prediction artifacts.

    Labs without a registered builder are skipped silently. Hidden, failed,
    and discarded lab results never become artifacts.
    """
    artifacts: list[PredictionArtifact] = []
    for result in report.selected:
        builder = _BUILDERS.get(result.lab_id)
        if builder is not None:
            artifacts.extend(builder(result, "selected"))
    for result in report.warning:
        builder = _BUILDERS.get(result.lab_id)
        if builder is not None:
            artifacts.extend(builder(result, "warning"))
    return artifacts
