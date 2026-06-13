from __future__ import annotations

import polars as pl

from app.text_engine.schemas import ExtractedTextSignal, TextDocumentSignal


def extract_text_signals(documents: list[TextDocumentSignal]) -> list[ExtractedTextSignal]:
    if not documents:
        return []
    frame = pl.DataFrame([_document_to_row(document) for document in documents], strict=False)
    scenario = str(frame.item(0, "scenario"))
    if scenario == "reputation_monitor":
        signals = _extract_reputation_signals(frame)
    else:
        signals = _extract_supply_chain_signals(frame)
    return [ExtractedTextSignal.model_validate(record) for record in signals.to_dicts()]


def _document_to_row(document: TextDocumentSignal) -> dict[str, object]:
    return {
        "dataset": "text_documents",
        "document_id": document.document_id,
        "scenario": document.scenario,
        "source_type": document.source_type,
        "source_name": document.source_name,
        "entity_name": document.entity_name,
        "entity_type": document.entity_type,
        "observed_at": document.observed_at,
        "url": document.url,
        "title": document.title,
        "text": document.text,
        "language": document.language,
        "period": document.metadata.get("period"),
        "time_bucket": document.metadata.get("time_bucket"),
        "price_drop_pct": document.metadata.get("price_drop_pct"),
        "sentiment_score": document.metadata.get("sentiment_score"),
        "lead_time_weeks": document.metadata.get("lead_time_weeks"),
        "price_change_pct": document.metadata.get("price_change_pct"),
        "predicted_cost_increase_pct": document.metadata.get("predicted_cost_increase_pct"),
        "delay_days": document.metadata.get("delay_days"),
        "risk_score": document.metadata.get("risk_score"),
        "price_premium_pct": document.metadata.get("price_premium_pct"),
        "compatibility_confidence": document.metadata.get("compatibility_confidence"),
    }


def _base_signal(frame: pl.DataFrame, *, signal_type: str, label: str, confidence: float, evidence_expr: pl.Expr, numeric_expr: pl.Expr | None = None, value_expr: pl.Expr | None = None) -> pl.DataFrame:
    return frame.select(
        pl.lit("text_signals").alias("dataset"),
        "document_id",
        "scenario",
        "source_type",
        "source_name",
        pl.col("entity_name").fill_null("unknown").alias("entity_name"),
        "entity_type",
        pl.lit(signal_type).alias("signal_type"),
        pl.lit(label).alias("label"),
        (value_expr if value_expr is not None else pl.lit(None, dtype=pl.Utf8)).alias("value"),
        (numeric_expr if numeric_expr is not None else pl.lit(None, dtype=pl.Float64)).alias("numeric_value"),
        pl.lit(confidence).alias("confidence"),
        evidence_expr.alias("evidence_text"),
        "observed_at",
        "period",
        "time_bucket",
    )


def _extract_reputation_signals(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("text").cast(pl.Utf8),
        pl.col("period").cast(pl.Utf8),
        pl.col("time_bucket").cast(pl.Utf8),
        pl.when(pl.col("sentiment_score").is_not_null()).then(pl.col("sentiment_score").cast(pl.Float64)).when(pl.col("text").str.contains("slow|wait|queue", literal=False)).then(pl.lit(-0.8)).when(pl.col("text").str.contains("good|pleasant|friendly|consistent", literal=False)).then(pl.lit(0.75)).otherwise(pl.lit(0.0)).alias("sentiment_numeric"),
    )
    sentiment = _base_signal(
        frame.filter(pl.col("source_type").is_in(["review", "social_mention"])),
        signal_type="sentiment",
        label="sentiment",
        confidence=0.90,
        evidence_expr=pl.col("text"),
        numeric_expr=pl.col("sentiment_numeric"),
        value_expr=pl.when(pl.col("sentiment_numeric") < 0).then(pl.lit("negative")).otherwise(pl.lit("positive")),
    )
    slow_service = _base_signal(
        frame.filter(pl.col("text").str.contains("slow", literal=False)),
        signal_type="complaint_topic",
        label="slow_service",
        confidence=0.88,
        evidence_expr=pl.lit("slow service"),
    )
    queue = _base_signal(
        frame.filter(pl.col("text").str.contains("wait|queue", literal=False)),
        signal_type="complaint_topic",
        label="queue_or_waiting",
        confidence=0.84,
        evidence_expr=pl.lit("waited or queue"),
    )
    morning_peak = _base_signal(
        frame.filter(pl.col("text").str.contains("morning|8-9 AM|8-9 am|8-9", literal=False) | (pl.col("time_bucket") == "morning")),
        signal_type="time_window",
        label="morning_peak",
        confidence=0.75,
        evidence_expr=pl.lit("this morning"),
    )
    competitor_discount = _base_signal(
        frame.filter((pl.col("source_type") == "menu_page") & pl.col("price_drop_pct").is_not_null()),
        signal_type="competitor_move",
        label="competitor_discount",
        confidence=0.90,
        evidence_expr=pl.col("text"),
        numeric_expr=pl.col("price_drop_pct").cast(pl.Float64),
        value_expr=(pl.col("price_drop_pct") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%"),
    )
    new_menu = _base_signal(
        frame.filter((pl.col("source_type") == "menu_page") & pl.col("text").str.contains("seasonal menu|new menu", literal=False)),
        signal_type="competitor_move",
        label="new_menu",
        confidence=0.81,
        evidence_expr=pl.col("text"),
    )
    oat_milk = _base_signal(
        frame.filter(pl.col("text").str.contains("oat milk", literal=False)),
        signal_type="menu_trend",
        label="oat_milk_trend",
        confidence=0.58,
        evidence_expr=pl.col("text"),
    )
    return pl.concat([sentiment, slow_service, queue, morning_peak, competitor_discount, new_menu, oat_milk], how="vertical_relaxed")


def _extract_supply_chain_signals(frame: pl.DataFrame) -> pl.DataFrame:
    lead_time = _base_signal(
        frame.filter(pl.col("lead_time_weeks").is_not_null()),
        signal_type="lead_time",
        label="lead_time_increase",
        confidence=0.92,
        evidence_expr=pl.col("text"),
        numeric_expr=pl.col("lead_time_weeks").cast(pl.Float64),
        value_expr=pl.col("lead_time_weeks").cast(pl.Int64).cast(pl.Utf8) + pl.lit(" weeks"),
    )
    commodity = _base_signal(
        frame.filter(pl.col("price_change_pct").is_not_null()),
        signal_type="price_change",
        label="commodity_price_pressure",
        confidence=0.90,
        evidence_expr=pl.col("text"),
        numeric_expr=(pl.col("price_change_pct") * 100).cast(pl.Float64),
        value_expr=(pl.col("price_change_pct") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%"),
    )
    battery = _base_signal(
        frame.filter(pl.col("predicted_cost_increase_pct").is_not_null()),
        signal_type="price_change",
        label="price_increase",
        confidence=0.87,
        evidence_expr=pl.col("text"),
        numeric_expr=(pl.col("predicted_cost_increase_pct") * 100).cast(pl.Float64),
        value_expr=(pl.col("predicted_cost_increase_pct") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%"),
    )
    shipping = _base_signal(
        frame.filter(pl.col("delay_days").is_not_null()),
        signal_type="shipping_delay",
        label="shipping_delay",
        confidence=0.86,
        evidence_expr=pl.col("text"),
        numeric_expr=pl.col("delay_days").cast(pl.Float64),
        value_expr=pl.col("delay_days").cast(pl.Int64).cast(pl.Utf8) + pl.lit(" days"),
    )
    alternative = _base_signal(
        frame.filter(pl.col("price_premium_pct").is_not_null()),
        signal_type="supplier_option",
        label="alternative_supplier_available",
        confidence=0.68,
        evidence_expr=pl.col("text"),
        numeric_expr=(pl.col("price_premium_pct") * 100).cast(pl.Float64),
        value_expr=(pl.col("price_premium_pct") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("% premium"),
    )
    geopolitical = _base_signal(
        frame.filter(pl.col("risk_score").is_not_null()),
        signal_type="geopolitical_risk",
        label="geopolitical_risk",
        confidence=0.61,
        evidence_expr=pl.col("text"),
        numeric_expr=pl.col("risk_score").cast(pl.Float64),
        value_expr=pl.col("risk_score").round(2).cast(pl.Utf8),
    )
    disruption = _base_signal(
        frame.filter(pl.col("text").str.contains("constrained|pricing pressure|delayed", literal=False)),
        signal_type="supply_risk",
        label="supplier_disruption",
        confidence=0.84,
        evidence_expr=pl.col("text"),
    )
    return pl.concat([lead_time, commodity, battery, shipping, alternative, geopolitical, disruption], how="vertical_relaxed")
