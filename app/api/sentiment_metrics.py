"""Per-location sentiment aggregation for the dashboard.

The LocationSentimentLab returns only the single worst-drop location; for
charts we need the full per-location current mean. We compute it directly
from the same `text_signals` the lab consumes, so the aggregation stays
consistent with lab inputs.
"""
from __future__ import annotations

import re

import polars as pl

from app.labs.schemas import LabContext


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def series_key(location_name: str) -> str:
    return f"{slug(location_name)}_sentiment"


def location_sentiment_snapshot(context: LabContext) -> dict[str, float]:
    """Return {series_key: mean_recent_sentiment} for every location with
    sentiment signals in the prepared context's `text_signals`.

    Uses period == "recent" so the chart reflects the latest observation
    window rather than the baseline.
    """
    signals = (
        context.scan_text_signals(
            columns=["entity_name", "entity_type", "signal_type", "numeric_value", "period"]
        )
        .filter(
            (pl.col("signal_type") == "sentiment")
            & (pl.col("entity_type") == "location")
            & (pl.col("period") == "recent")
        )
        .with_columns(
            pl.col("entity_name").cast(pl.Utf8),
            pl.col("numeric_value").cast(pl.Float64),
        )
        .group_by("entity_name")
        .agg(pl.col("numeric_value").mean().alias("avg_sentiment"))
        .sort("entity_name")
        .collect()
    )

    snapshot: dict[str, float] = {}
    for row in signals.iter_rows(named=True):
        entity = row["entity_name"]
        value = row["avg_sentiment"]
        if entity is None or value is None:
            continue
        snapshot[series_key(str(entity))] = round(float(value), 4)
    return snapshot


def location_series_specs(snapshot: dict[str, float]) -> list[tuple[str, str]]:
    """Return [(series_key, human_label)] derived from a snapshot.

    Label heuristic: take the series_key, strip the trailing `_sentiment`,
    title-case the remaining underscore-separated words.
    """
    pairs: list[tuple[str, str]] = []
    for key in sorted(snapshot.keys()):
        stem = key[: -len("_sentiment")] if key.endswith("_sentiment") else key
        label = " ".join(part.capitalize() for part in stem.split("_") if part)
        pairs.append((key, label or key))
    return pairs
