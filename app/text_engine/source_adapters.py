from __future__ import annotations

from pathlib import Path

import polars as pl

from app.text_engine.schemas import TextDocumentSignal


def load_raw_external_records(path: Path) -> list[dict[str, object]]:
    return pl.read_ndjson(path).to_dicts()


def adapt_raw_records_to_documents(scenario: str, raw_records: list[dict[str, object]]) -> list[TextDocumentSignal]:
    adapters = {
        ("reputation_monitor", "raw_review"): _adapt_reputation_review,
        ("reputation_monitor", "raw_social_mention"): _adapt_reputation_social,
        ("reputation_monitor", "raw_menu_page"): _adapt_reputation_menu,
        ("supply_chain_risk", "raw_supplier_page"): _adapt_supply_supplier,
        ("supply_chain_risk", "raw_shipping_update"): _adapt_supply_shipping,
        ("supply_chain_risk", "raw_commodity_report"): _adapt_supply_commodity,
        ("supply_chain_risk", "raw_news"): _adapt_supply_news,
    }
    documents: list[TextDocumentSignal] = []
    for record in raw_records:
        adapter = adapters[(scenario, str(record["dataset"]))]
        documents.append(adapter(record))
    return documents


def _base_document(record: dict[str, object], *, scenario: str, source_type: str) -> dict[str, object]:
    return {
        "document_id": str(record["document_id"]),
        "scenario": scenario,
        "source_type": source_type,
        "source_name": str(record["source_name"]),
        "entity_name": record.get("entity_name"),
        "entity_type": record.get("entity_type"),
        "observed_at": record.get("observed_at"),
        "url": record.get("url"),
        "title": record.get("title"),
        "text": str(record["text"]),
        "language": record.get("language"),
        "metadata": {
            key: value
            for key, value in record.items()
            if key
            not in {
                "dataset",
                "document_id",
                "source_name",
                "entity_name",
                "entity_type",
                "observed_at",
                "url",
                "title",
                "text",
                "language",
            }
        },
    }


def _adapt_reputation_review(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="reputation_monitor", source_type="review"))


def _adapt_reputation_social(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="reputation_monitor", source_type="social_mention"))


def _adapt_reputation_menu(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="reputation_monitor", source_type="menu_page"))


def _adapt_supply_supplier(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="supply_chain_risk", source_type="supplier_page"))


def _adapt_supply_shipping(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="supply_chain_risk", source_type="shipping_update"))


def _adapt_supply_commodity(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="supply_chain_risk", source_type="commodity_report"))


def _adapt_supply_news(record: dict[str, object]) -> TextDocumentSignal:
    return TextDocumentSignal.model_validate(_base_document(record, scenario="supply_chain_risk", source_type="news"))
