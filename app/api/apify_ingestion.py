"""Normalize Apify-fetched review items into the canonical raw_review shape.

The lab pipeline (``app.text_engine.source_adapters``) consumes records
with the schema documented in
``app/demo_data/reputation_monitor_external_raw.ndjson``::

    {
      "dataset": "raw_review",
      "document_id": "...",
      "source_name": "...",
      "entity_name": "Miners <location>",
      "entity_type": "location",
      "observed_at": "ISO-8601",
      "url": "...",
      "text": "...",
      "language": "en",
      "period": "recent",
      "time_bucket": "morning|midday|afternoon|evening|other",
      "sentiment_score": float (0..1, optional)
    }

This module is the **only** place Apify-shaped rows are converted to the
canonical shape. Keeping the adapter narrow and pure means the labs
themselves never see the Apify schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# How a 1..5 star rating maps to a normalized sentiment_score in 0..1.
# This is a deterministic prior, not a model — labs already re-derive
# sentiment via the text_engine, so we just need a reasonable starting
# value for sources that don't carry one (e.g. Google Reviews).
_STAR_TO_SENTIMENT: dict[int, float] = {
    1: 0.15,
    2: 0.35,
    3: 0.55,
    4: 0.75,
    5: 0.90,
}


def _time_bucket(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "other"


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Apify returns trailing Z; fromisoformat handles +00:00, swap it.
        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _coerce_str(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_sentiment(stars: int | None) -> float | None:
    if stars is None:
        return None
    return _STAR_TO_SENTIMENT.get(stars)


def _resolve_entity(item: dict[str, Any]) -> str | None:
    """Pick the location name from the Apify item.

    Google Reviews actor populates ``placeName``; we also accept
    ``entity_name`` so consumers can preformat. Returns None if neither
    is present — caller will drop the record.
    """
    for key in ("placeName", "entity_name", "title"):
        value = item.get(key)
        if value:
            return str(value).strip()
    return None


def normalize_apify_reviews(
    items: list[dict[str, Any]],
    *,
    source_name: str = "Apify: Google Reviews",
    default_period: str = "recent",
) -> list[dict[str, object]]:
    """Convert raw Apify review items to canonical raw_review records.

    Skips items missing critical fields (text or a usable entity name)
    rather than fabricating data — invalid rows return as zero output,
    not as junk records the labs would have to filter again.
    """
    canonical: list[dict[str, object]] = []
    for index, item in enumerate(items):
        text = _coerce_str(item.get("text"))
        entity = _resolve_entity(item)
        if not text or not entity:
            continue

        observed = _parse_iso(item.get("publishedAtDate") or item.get("observed_at"))
        observed_iso: str
        time_bucket: str
        if observed is None:
            # Stable placeholder so downstream sorting stays deterministic.
            # Match the demo fixture's naive-UTC ISO format so Polars can
            # infer a single datetime supertype when records are merged.
            observed_iso = "2026-01-01T00:00:00"
            time_bucket = "other"
        else:
            if observed.tzinfo is not None:
                observed = observed.astimezone(timezone.utc).replace(tzinfo=None)
            observed_iso = observed.isoformat()
            time_bucket = _time_bucket(observed.hour)

        stars = _coerce_int(item.get("stars"))
        sentiment = _coerce_sentiment(stars)

        document_id = _coerce_str(item.get("reviewId")) or f"apify-review-{index:03d}"

        canonical.append(
            {
                "dataset": "raw_review",
                "document_id": document_id,
                "source_name": _coerce_str(item.get("source_name"), default=source_name)
                or source_name,
                "entity_name": entity,
                "entity_type": "location",
                "observed_at": observed_iso,
                "url": _coerce_str(item.get("url")) or None,
                "text": text,
                "language": _coerce_str(item.get("language"), default="en") or "en",
                "period": _coerce_str(item.get("period"), default=default_period) or default_period,
                "time_bucket": time_bucket,
                "sentiment_score": sentiment,
                # Apify-origin marker so /refresh metadata can prove the
                # pipeline actually merged external data.
                "ingested_via": "apify",
                "stars": stars,
            }
        )
    return canonical
