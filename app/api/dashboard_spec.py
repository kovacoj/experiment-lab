"""Build DashboardSpec from a MonitoringPlan plus the latest sentiment snapshot.

The slice supports exactly one chart_id: `sentiment_trend_by_location`. New
charts must be added here explicitly — there is no generic chart-from-artifact
auto-builder. That keeps chart contracts visible and testable.
"""
from __future__ import annotations

from app.api.schemas import (
    ChartSeriesSpec,
    ChartSpec,
    DashboardSpec,
    MonitoringPlan,
)
from app.api.sentiment_metrics import location_series_specs

CHART_SENTIMENT_TREND = "sentiment_trend_by_location"


def _build_sentiment_trend_chart(
    scenario: str,
    session_id: str,
    snapshot: dict[str, float],
    source_model_ids: list[str],
    source_lab_ids: list[str],
) -> ChartSpec:
    series = [
        ChartSeriesSpec(key=key, label=label)
        for key, label in location_series_specs(snapshot)
    ]
    return ChartSpec(
        chart_id=CHART_SENTIMENT_TREND,
        scenario=scenario,
        title="Sentiment trend by location",
        description="Mean recent-review sentiment per location, one point per refresh.",
        type="line",
        x_key="time",
        series=series,
        data_endpoint=f"/sessions/{session_id}/charts/{CHART_SENTIMENT_TREND}/data",
        refresh_interval_ms=10000,
        source_model_ids=source_model_ids,
        source_lab_ids=source_lab_ids,
        empty_state="No refreshes yet — trigger a refresh to populate this chart.",
    )


def build_dashboard_spec(
    session_id: str,
    scenario: str,
    title: str,
    plan: MonitoringPlan,
    sentiment_snapshot: dict[str, float],
    last_updated: str,
    headline: str | None,
) -> DashboardSpec:
    charts: list[ChartSpec] = []
    for model in plan.monitored_models:
        for chart_id in model.chart_ids:
            if chart_id == CHART_SENTIMENT_TREND:
                charts.append(
                    _build_sentiment_trend_chart(
                        scenario=scenario,
                        session_id=session_id,
                        snapshot=sentiment_snapshot,
                        source_model_ids=[model.model_id],
                        source_lab_ids=[model.lab_id],
                    )
                )
            # Unknown chart_ids are skipped silently; add a builder when adding
            # a chart. We deliberately do not synthesize a placeholder.
    return DashboardSpec(
        session_id=session_id,
        scenario=scenario,
        title=title,
        headline=headline,
        last_updated=last_updated,
        charts=charts,
    )
