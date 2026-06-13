from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class LocationSentimentLab(BaseLab):
    lab_id = "location_sentiment"
    lab_name = "Location Sentiment Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        locations = context.scan_internal("locations", ["location_id", "location_name"]).with_columns(
            pl.col("location_id").cast(pl.Utf8),
            pl.col("location_name").cast(pl.Utf8),
        )
        review_metrics = context.scan_text_signals(columns=["entity_name", "entity_type", "signal_type", "numeric_value", "period", "document_id"])
        review_metrics = review_metrics.filter((pl.col("signal_type") == "sentiment") & (pl.col("entity_type") == "location")).with_columns(
            pl.col("entity_name").cast(pl.Utf8),
            pl.col("period").cast(pl.Utf8),
            pl.col("numeric_value").cast(pl.Float32),
        )
        grouped = review_metrics.group_by(["entity_name", "period"]).agg(
            pl.col("numeric_value").mean().alias("avg_sentiment"),
            pl.n_unique("document_id").alias("review_count"),
        )
        baseline = grouped.filter(pl.col("period") == "baseline").select(
            "entity_name",
            pl.col("avg_sentiment").alias("baseline_sentiment"),
        )
        recent = grouped.filter(pl.col("period") == "recent").select(
            "entity_name",
            pl.col("avg_sentiment").alias("recent_sentiment"),
            pl.col("review_count").alias("recent_review_count"),
        )
        revenue_grouped = (
            context.scan_internal("revenue", ["location_id", "period", "revenue_index"])
            .with_columns(
                pl.col("location_id").cast(pl.Utf8),
                pl.col("period").cast(pl.Utf8),
                pl.col("revenue_index").cast(pl.Float32),
            )
            .join(locations, on="location_id", how="left")
            .group_by(["location_name", "period"])
            .agg(pl.col("revenue_index").mean().alias("avg_revenue_index"))
        )
        revenue_baseline = revenue_grouped.filter(pl.col("period") == "baseline").select(
            "location_name",
            pl.col("avg_revenue_index").alias("baseline_revenue"),
        )
        revenue_recent = revenue_grouped.filter(pl.col("period") == "recent").select(
            "location_name",
            pl.col("avg_revenue_index").alias("recent_revenue"),
        )
        best_row_frame = (
            baseline.join(recent, on="entity_name", how="inner")
            .join(revenue_baseline, left_on="entity_name", right_on="location_name", how="left")
            .join(revenue_recent, left_on="entity_name", right_on="location_name", how="left")
            .with_columns(
                (pl.col("baseline_sentiment") - pl.col("recent_sentiment")).alias("sentiment_drop"),
                (pl.col("recent_revenue") - pl.col("baseline_revenue")).alias("revenue_change"),
            )
            .sort("sentiment_drop", descending=True)
            .limit(1)
            .collect()
        )

        if best_row_frame.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="A location-specific sentiment drop indicates an operational issue that may affect revenue.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No comparable baseline and recent sentiment data was available.",
                evidence=[],
                limitations=["Missing review history by location."],
            )

        best_row = best_row_frame.row(0, named=True)
        best_location_name = str(best_row["entity_name"])
        best_drop = float(best_row["sentiment_drop"])
        revenue_change = float(best_row["revenue_change"])
        location_name = best_location_name
        top_topic_frame = (
            context.scan_text_signals(columns=["entity_name", "period", "signal_type", "label"])
            .filter(
                (pl.col("entity_name") == best_location_name)
                & (pl.col("period") == "recent")
                & (pl.col("signal_type") == "complaint_topic")
            )
            .group_by("label")
            .agg(pl.len().alias("topic_count"))
            .sort("topic_count", descending=True)
            .limit(1)
            .collect()
        )
        top_topic = "service quality"
        if not top_topic_frame.is_empty():
            top_topic = str(top_topic_frame.item(0, "label")).replace("_", " ")

        score = clamp(best_drop * 3.0 + (0.10 if revenue_change < 0 else 0.0) + 0.10)
        confidence = 0.83
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="A location-specific sentiment drop indicates an operational issue that may affect revenue.",
            status=status,
            score=score,
            confidence=confidence,
            summary=f"{location_name} sentiment dropped by {round(best_drop * 100):d}% with {top_topic} complaints clustering in recent feedback.",
            evidence=[
                EvidenceItem(source="external", label="recent_reviews", value=int(best_row["recent_review_count"]), detail=f"Recent {location_name} reviews mention slow service and waiting."),
                EvidenceItem(source="internal", label="revenue_change_index", value=round(revenue_change, 2), detail=f"{location_name} revenue softened versus baseline."),
                EvidenceItem(source="derived", label="sentiment_drop", value=round(best_drop, 2), detail=f"Recent sentiment is {round(best_drop * 100):d}% below baseline."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Investigate morning operations",
                    detail=f"Investigate {location_name} morning operations and monitor sentiment recovery for 3 days.",
                    urgency="high",
                )
            ],
            limitations=["Sentiment is inferred from cached demo review scores rather than live platform retrieval."],
            monitoring_rules=[{"type": "sentiment_drop", "entity_name": best_location_name, "threshold": 0.15}],
        )
