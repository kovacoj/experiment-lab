from __future__ import annotations

import polars as pl


def aggregate_signals_by_entity(signals: pl.LazyFrame, *, signal_type: str, label: str | None = None) -> pl.LazyFrame:
    frame = signals.filter(pl.col("signal_type") == signal_type)
    if label is not None:
        frame = frame.filter(pl.col("label") == label)
    return frame.group_by(["entity_name", "period", "time_bucket"]).agg(
        pl.len().alias("signal_count"),
        pl.mean("numeric_value").alias("avg_numeric_value"),
        pl.mean("confidence").alias("avg_confidence"),
    )
