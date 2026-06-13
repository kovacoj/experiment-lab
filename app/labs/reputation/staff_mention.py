from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class StaffMentionLab(BaseLab):
    lab_id = "staff_mention"
    lab_name = "Staff Mention Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        morning_schedule_count = int(
            context.scan_internal("staff_schedule", ["hour"])
            .with_columns(pl.col("hour").cast(pl.Int8))
            .filter(pl.col("hour").is_in([8, 9]))
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        sparse = morning_schedule_count < 4
        morning_signal = int(
            context.scan_text_signals(columns=["time_bucket", "signal_type", "label"])
            .filter(
                (pl.col("time_bucket") == "morning")
                & (pl.col("signal_type") == "complaint_topic")
                & (pl.col("label").is_in(["slow_service", "queue_or_waiting"]))
            )
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        score = clamp(0.20 + min(morning_signal / 20.0, 0.15))
        confidence = 0.49 if sparse else 0.55
        status = default_status(score, confidence)

        limitations = ["Insufficient person-level evidence is available in the schedule and review data."]
        if sparse:
            limitations.append("Morning shift coverage is partial, so attribution stays shift-level only.")

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="If complaints cluster during a specific shift, the issue may be operational.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Staff schedule data is too sparse for individual attribution. Morning shift correlation is visible, but not enough for person-level claims.",
            evidence=[
                EvidenceItem(source="internal", label="morning_schedule_records", value=morning_schedule_count, detail="Only partial shift coverage is available for the relevant morning window."),
                EvidenceItem(source="external", label="morning_complaint_mentions", value=morning_signal, detail="Complaints cluster around the morning shift rather than a named employee."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Monitor by shift, not person",
                    detail="Use shift-level monitoring and improve schedule completeness before making stronger operational claims.",
                    urgency="medium",
                )
            ],
            limitations=limitations,
            monitoring_rules=[{"type": "shift_level_only", "time_bucket": "morning"}],
        )
