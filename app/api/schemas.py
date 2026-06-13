"""API-layer Pydantic contracts.

Subset of the full MonitoringPlan/Dashboard spec described in the handoff —
only the fields the current vertical slice exercises. Add fields as new
slices need them; do not pre-add speculative fields.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ArtifactStatus = Literal["selected", "warning", "hidden", "discarded", "failed"]
Priority = Literal["primary", "secondary", "exploration"]
ChartType = Literal["line", "bar", "horizontal_bar", "area", "event_timeline"]
Severity = Literal["info", "warning", "critical"]


class PredictionArtifact(BaseModel):
    model_id: str
    lab_id: str
    scenario: str
    prediction_name: str
    entity_scope: str | None = None
    metric_keys: list[str] = Field(default_factory=list)
    output_metric: str
    current_value: float | int | str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: ArtifactStatus
    monitoring_eligible: bool = False
    chart_eligible: bool = False
    alert_eligible: bool = False
    recommended_refresh_minutes: int = 60
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitoredModelSpec(BaseModel):
    model_id: str
    lab_id: str
    scenario: str
    prediction_name: str
    entity_scope: str | None = None
    refresh_minutes: int
    chart_ids: list[str] = Field(default_factory=list)
    alert_policy_id: str | None = None
    enabled: bool = True
    priority: Priority = "primary"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitoringPlan(BaseModel):
    monitoring_plan_id: str
    session_id: str
    scenario: str
    enabled: bool = True
    monitored_models: list[MonitoredModelSpec] = Field(default_factory=list)
    refresh_endpoint: str
    dashboard_endpoint: str
    alert_endpoint: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartSeriesSpec(BaseModel):
    key: str
    label: str


class ChartThresholdSpec(BaseModel):
    key: str
    value: float
    label: str
    severity: Severity = "warning"


class ChartSpec(BaseModel):
    chart_id: str
    scenario: str
    title: str
    description: str
    type: ChartType
    x_key: str
    series: list[ChartSeriesSpec]
    data_endpoint: str
    refresh_interval_ms: int = 10000
    thresholds: list[ChartThresholdSpec] = Field(default_factory=list)
    source_lab_ids: list[str] = Field(default_factory=list)
    source_model_ids: list[str] = Field(default_factory=list)
    empty_state: str = "No data yet."
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardSpec(BaseModel):
    session_id: str
    scenario: str
    title: str
    headline: str | None = None
    last_updated: str | None = None
    cards: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertEnvelope(BaseModel):
    """Slice stub: alert policy is deferred. Always emits should_notify=false."""

    should_notify: bool = False
    title: str | None = None
    body: str | None = None
    severity: Severity | None = None
    recommended_action: str | None = None
    dedupe_key: str | None = None


class DashboardDelta(BaseModel):
    updated_chart_ids: list[str] = Field(default_factory=list)
    last_updated: str


class RefreshRequest(BaseModel):
    source: str = "manual"
    mode: Literal["demo", "live"] = "demo"
    scenario: str | None = None
    monitoring_plan_id: str | None = None
    streams: dict[str, Any] = Field(default_factory=dict)


class RefreshResponse(BaseModel):
    session_id: str
    scenario: str
    prediction_changed: bool
    models: list[dict[str, Any]]
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    alert: AlertEnvelope
    decision_cards: list[dict[str, Any]] = Field(default_factory=list)
    dashboard_delta: DashboardDelta
    # Provenance for any external (Apify) stream merged into this refresh.
    # Absent when streams.external was not requested.
    external_provenance: dict[str, Any] | None = None
    # Most recently logged operator notes for the session. Absent if the
    # operator has not posted any context. Treated as a low-confidence
    # supplementary signal — never as ground truth.
    user_context: dict[str, Any] | None = None


class MonitoringPlanBuildRequest(BaseModel):
    source: str = "manual"
    scenario: str | None = None


class ChartDataResponse(BaseModel):
    chart_id: str
    data: list[dict[str, Any]]


class UserContextEntry(BaseModel):
    """A single operator-supplied note attached to a session.

    Stored append-only in tmp/sessions/{id}/user_context.json. Surfaces
    in /refresh metadata as a low-confidence supplementary signal — the
    labs themselves do not consume it as ground truth.
    """
    message: str = Field(min_length=1, max_length=2000)
    source: str = Field(default="manual", max_length=64)
    tags: list[str] = Field(default_factory=list)


class UserContextCreated(BaseModel):
    session_id: str
    entry_id: str
    message: str
    source: str
    tags: list[str]
    created_at: str
    total_entries: int


class UserContextListResponse(BaseModel):
    session_id: str
    count: int
    entries: list[dict[str, Any]]
