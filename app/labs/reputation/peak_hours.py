from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class PeakHoursAnalysisLab(BaseLab):
    lab_id = "peak_hours"
    lab_name = "Peak Hours Analysis Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        locations = context.scan_internal("locations", ["location_id", "location_name"]).with_columns(
            pl.col("location_id").cast(pl.Categorical),
            pl.col("location_name").cast(pl.Utf8),
        )
        transactions = context.scan_internal("transactions_by_hour", ["location_id", "hour", "transactions"]).with_columns(
            pl.col("location_id").cast(pl.Categorical),
            pl.col("hour").cast(pl.Int8),
            pl.col("transactions").cast(pl.Int16),
        )
        schedules = context.scan_internal("staff_schedule", ["location_id", "hour", "staff_count"]).with_columns(
            pl.col("location_id").cast(pl.Categorical),
            pl.col("hour").cast(pl.Int8),
            pl.col("staff_count").cast(pl.Int8),
        )
        overload = (
            transactions.join(schedules, on=["location_id", "hour"], how="inner")
            .join(locations, on="location_id", how="left")
            .with_columns((pl.col("transactions") / pl.col("staff_count").clip(lower_bound=1)).alias("transactions_per_staff"))
            .sort("transactions_per_staff", descending=True)
            .limit(1)
            .collect()
        )

        if overload.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="Complaints during peak hours indicate a staffing or capacity mismatch.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No hourly transaction data was available.",
                evidence=[],
                limitations=["Missing transactions by location and hour."],
            )

        best_record = overload.row(0, named=True)
        location_id = str(best_record["location_id"])
        hour = int(best_record["hour"])
        location_name = str(best_record["location_name"])
        best_load = float(best_record["transactions_per_staff"])
        queue_mentions = int(
            context.scan_text_signals(columns=["entity_name", "period", "signal_type", "label"])
            .filter(
                (pl.col("entity_name") == location_name)
                & (pl.col("period") == "recent")
                & (pl.col("signal_type") == "complaint_topic")
                & (pl.col("label").is_in(["slow_service", "queue_or_waiting"]))
            )
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        score = clamp(0.45 + min(best_load / 80.0, 0.25) + min(queue_mentions / 20.0, 0.10))
        confidence = 0.72
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Complaints during peak hours indicate a staffing or capacity mismatch.",
            status=status,
            score=score,
            confidence=confidence,
            summary=f"{location_name} shows queue pressure around {hour}-{hour + 1} AM, suggesting a peak-hour staffing mismatch.",
            evidence=[
                EvidenceItem(source="internal", label="transactions_per_staff", value=round(best_load, 1), detail=f"{location_name} handled heavy transaction volume per staff member during the morning peak."),
                EvidenceItem(source="external", label="queue_mentions", value=queue_mentions, detail="Recent reviews mention waiting, queue, or slow service during the morning rush."),
                EvidenceItem(source="derived", label="peak_hour", value=f"{hour}-{hour + 1} AM", detail="The highest overload window is concentrated in the morning peak."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Add one extra morning staff member",
                    detail=f"Add one extra staff member during {hour}-{hour + 1} AM at {location_name} and monitor review recovery.",
                    urgency="high",
                )
            ],
            limitations=["Queue pressure is inferred from transactions-per-staff and review text rather than direct wait-time telemetry."],
            monitoring_rules=[{"type": "transactions_per_staff", "entity_name": location_name, "threshold": 25}],
        )
