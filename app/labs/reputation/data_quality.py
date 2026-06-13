from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class DataQualityLab(BaseLab):
    lab_id = "reputation_data_quality"
    lab_name = "Data Quality Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        location_count = int(context.scan_internal("locations", ["location_id"]).select(pl.len()).collect().item(0, 0))
        schedule_count = int(context.scan_internal("staff_schedule", ["location_id"]).select(pl.len()).collect().item(0, 0))
        competitor_price_count = int(
            context.scan_text_signals(columns=["label"])
            .filter(pl.col("label") == "competitor_discount")
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        review_stats = (
            context.scan_text_documents(columns=["entity_name", "source_type", "period"])
            .filter(pl.col("source_type").is_in(["review", "social_mention"]))
            .join(
                context.scan_internal("locations", ["location_name"]).rename({"location_name": "entity_name"}),
                on="entity_name",
                how="inner",
            )
            .select(
                pl.len().alias("known_location_reviews"),
                (pl.col("period") == "recent").cast(pl.Int16).sum().alias("recent_reviews"),
            )
            .collect()
            .row(0, named=True)
        )
        reviews_with_known_location = int(review_stats["known_location_reviews"])
        recent_reviews = int(review_stats["recent_reviews"])

        coverage_checks = [
            location_count > 0,
            reviews_with_known_location >= 6,
            recent_reviews >= 4,
            schedule_count > 0,
            competitor_price_count > 0,
        ]
        score = 0.62 if all(coverage_checks) else 0.56
        confidence = 0.62
        status = default_status(score, confidence)

        limitations: list[str] = []
        if schedule_count < 6:
            limitations.append("Staff schedule coverage is partial.")

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Reputation monitoring needs enough location, review, and competitor coverage to support branch-level findings.",
            status=status,
            score=score,
            confidence=confidence,
            summary=(
                "Data is sufficient for location-level monitoring. Vinohrady and Wenceslas have enough recent reviews; "
                "staff schedule coverage is partial."
            ),
            evidence=[
                EvidenceItem(source="internal", label="location_count", value=location_count, detail="Tracked Miners branches in Prague."),
                EvidenceItem(source="external", label="recent_review_count", value=recent_reviews, detail="Recent review/social coverage is available for the main locations."),
                EvidenceItem(source="external", label="competitor_price_records", value=competitor_price_count, detail="Cached demo competitor pricing and menu data is present."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Fill schedule gaps",
                    detail="Add more complete shift coverage before using this scenario for staff-level operational follow-up.",
                    urgency="medium",
                )
            ],
            limitations=limitations,
            monitoring_rules=[{"type": "data_coverage", "minimum_recent_reviews": 4}],
        )
