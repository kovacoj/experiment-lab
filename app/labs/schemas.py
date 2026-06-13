from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field, model_validator


ScenarioName = Literal["reputation_monitor", "supply_chain_risk"]
LabStatus = Literal["selected", "hidden", "warning", "failed", "inconclusive", "discarded"]
Urgency = Literal["low", "medium", "high", "critical"]


class DataSource(BaseModel):
    records: list[dict[str, Any]] | None = None
    path: Path | None = None
    format: Literal["ndjson"] = "ndjson"

    @model_validator(mode="after")
    def validate_source(self) -> DataSource:
        if self.records is None and self.path is None:
            raise ValueError("DataSource requires either records or path.")
        return self

    def scan(self) -> pl.LazyFrame:
        if self.path is not None:
            return pl.scan_ndjson(self.path)
        if not self.records:
            return pl.DataFrame(schema={"dataset": pl.Utf8}).lazy()
        return pl.LazyFrame(self.records or [], strict=False)

    def scan_dataset(self, dataset: str, columns: list[str] | tuple[str, ...] | None = None) -> pl.LazyFrame:
        frame = self.scan().filter(pl.col("dataset") == dataset)
        if columns:
            frame = frame.select(list(columns))
        return frame

    def count_rows(self) -> int:
        return int(self.scan().select(pl.len().alias("row_count")).collect().item(0, 0))


class EvidenceItem(BaseModel):
    source: Literal["internal", "external", "derived"]
    label: str
    value: str | float | int | None = None
    detail: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RecommendedAction(BaseModel):
    title: str
    detail: str
    urgency: Urgency = "medium"


class LabContext(BaseModel):
    scenario: ScenarioName
    internal_data: DataSource
    external_data: DataSource
    text_documents: DataSource | None = None
    text_signals: DataSource | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    analysis_contract: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        scenario: ScenarioName,
        internal_data: list[dict[str, Any]],
        external_data: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        analysis_contract: dict[str, Any] | None = None,
    ) -> LabContext:
        return cls(
            scenario=scenario,
            internal_data=DataSource(records=internal_data),
            external_data=DataSource(records=external_data),
            text_documents=DataSource(records=[]),
            text_signals=DataSource(records=[]),
            metadata=metadata or {},
            analysis_contract=analysis_contract or {},
        )

    def scan_internal(self, dataset: str, columns: list[str] | tuple[str, ...] | None = None) -> pl.LazyFrame:
        return self.internal_data.scan_dataset(dataset, columns)

    def scan_external(self, dataset: str, columns: list[str] | tuple[str, ...] | None = None) -> pl.LazyFrame:
        return self.external_data.scan_dataset(dataset, columns)

    def scan_text_documents(self, dataset: str = "text_documents", columns: list[str] | tuple[str, ...] | None = None) -> pl.LazyFrame:
        source = self.text_documents or DataSource(records=[])
        return source.scan_dataset(dataset, columns)

    def scan_text_signals(self, dataset: str = "text_signals", columns: list[str] | tuple[str, ...] | None = None) -> pl.LazyFrame:
        source = self.text_signals or DataSource(records=[])
        return source.scan_dataset(dataset, columns)


class LabResult(BaseModel):
    lab_id: str
    lab_name: str
    scenario: ScenarioName
    hypothesis: str
    status: LabStatus
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[EvidenceItem]
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    output_cards: list[dict[str, Any]] = Field(default_factory=list)
    monitoring_rules: list[dict[str, Any]] = Field(default_factory=list)
    signal_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    business_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    actionability: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty: float | None = Field(default=None, ge=0.0, le=1.0)
    data_quality_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    duplication_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    safety_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    exploitation_score: float | None = Field(default=None)
    exploration_bonus: float | None = Field(default=None)
    final_priority_score: float | None = Field(default=None)
    reason_for_selection: str | None = None
    reason_for_hiding: str | None = None
    reason_for_discarding: str | None = None


class EnsembleFinding(BaseModel):
    ensemble_id: str
    scenario: ScenarioName
    title: str
    summary: str
    status: Literal["selected", "warning", "hidden"]
    contributing_lab_ids: list[str]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    final_priority_score: float = 0.0
    reason_for_selection: str | None = None


class DecisionCard(BaseModel):
    card_id: str
    scenario: ScenarioName
    title: str
    card_type: Literal["ensemble", "finding", "warning"]
    summary: str
    priority: float = 0.0
    status: Literal["selected", "warning", "hidden"]
    supporting_lab_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class LabRunReport(BaseModel):
    scenario: ScenarioName
    selected: list[LabResult]
    warning: list[LabResult]
    hidden: list[LabResult]
    failed: list[LabResult]
    discarded: list[LabResult]
    ensembles: list[EnsembleFinding]
    executive_summary: str
