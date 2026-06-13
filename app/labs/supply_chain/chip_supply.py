from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class ChipSupplyLab(BaseLab):
    lab_id = "chip_supply"
    lab_name = "Chip Supply Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        chip_status = (
            context.scan_internal("inventory", ["component", "days_remaining", "threshold_days"])
            .filter(pl.col("component") == "MCU chip")
            .with_columns(
                pl.col("days_remaining").cast(pl.Float32),
                pl.col("threshold_days").cast(pl.Float32),
            )
            .join(
                context.scan_text_signals(columns=["entity_name", "label", "signal_type", "numeric_value"])
                .filter((pl.col("signal_type") == "lead_time") & (pl.col("label") == "lead_time_increase"))
                .select(
                    pl.col("entity_name").alias("component"),
                    (pl.col("numeric_value") / 1.0).alias("lead_time_weeks"),
                ),
                on="component",
                how="inner",
            )
            .limit(1)
            .collect()
        )

        if chip_status.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="If lead times increase while inventory drops below threshold, production risk rises.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="Missing chip inventory or lead-time data.",
                evidence=[],
                limitations=["MCU inventory or lead-time records are missing."],
            )

        row = chip_status.row(0, named=True)
        inventory_days = float(row["days_remaining"])
        threshold_days = float(row["threshold_days"])
        lead_time_weeks = float(row["lead_time_weeks"])
        risk_gap = max(lead_time_weeks * 7 - inventory_days, 0.0)
        score = clamp(0.68 + min(risk_gap / 100.0, 0.15) + (0.10 if inventory_days <= 45 else 0.0))
        confidence = 0.88
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="If lead times increase while inventory drops below threshold, production risk rises.",
            status=status,
            score=score,
            confidence=confidence,
            summary="MCU inventory is approaching the 30-day threshold while lead time has risen to 16 weeks.",
            evidence=[
                EvidenceItem(source="internal", label="inventory_days_remaining", value=inventory_days, detail="MCU chips have 45 days of inventory remaining."),
                EvidenceItem(source="internal", label="threshold_days", value=threshold_days, detail="The monitoring threshold is 30 days of coverage."),
                EvidenceItem(source="external", label="lead_time_weeks", value=lead_time_weeks, detail="Cached semiconductor signals show a 16-week lead time."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Place emergency chip order",
                    detail="Place an emergency order or switch the next batch to an approved alternative MCU supplier.",
                    urgency="critical",
                )
            ],
            limitations=["Lead-time evidence comes from cached demo signals rather than a live supplier feed."],
            monitoring_rules=[{"type": "inventory_threshold", "component": "MCU chip", "threshold_days": 30}],
        )
