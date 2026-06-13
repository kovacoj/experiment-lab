from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextDocumentSignal(BaseModel):
    document_id: str
    scenario: Literal["reputation_monitor", "supply_chain_risk"]
    source_type: Literal[
        "review",
        "social_mention",
        "website",
        "menu_page",
        "news",
        "supplier_page",
        "shipping_update",
        "commodity_report",
    ]
    source_name: str
    entity_name: str | None
    entity_type: str | None
    observed_at: datetime | None
    url: str | None = None
    title: str | None = None
    text: str
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedTextSignal(BaseModel):
    document_id: str
    scenario: str
    source_type: str
    source_name: str
    entity_name: str
    entity_type: str | None
    signal_type: str
    label: str
    value: float | str | None = None
    numeric_value: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str
    observed_at: datetime | None = None
    period: str | None = None
    time_bucket: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
