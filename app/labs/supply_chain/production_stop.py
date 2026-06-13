from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class ProductionStopRiskLab(BaseLab):
    lab_id = "production_stop_risk"
    lab_name = "Production Stop Risk Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        risk_inputs = (
            context.scan_internal("inventory", ["component", "days_remaining", "threshold_days"])
            .filter(pl.col("component") == "MCU chip")
            .with_columns(
                pl.col("days_remaining").cast(pl.Float32),
                pl.col("threshold_days").cast(pl.Float32),
            )
            .join(
                context.scan_text_signals(columns=["entity_name", "signal_type", "label", "numeric_value"])
                .filter((pl.col("signal_type") == "lead_time") & (pl.col("label") == "lead_time_increase"))
                .select(
                    pl.col("entity_name").alias("component"),
                    pl.col("numeric_value").cast(pl.Float32).alias("lead_time_weeks"),
                ),
                on="component",
                how="inner",
            )
            .join(
                context.scan_text_signals(columns=["signal_type", "label", "numeric_value"])
                .filter((pl.col("signal_type") == "shipping_delay") & (pl.col("label") == "shipping_delay"))
                .select(pl.col("numeric_value").cast(pl.Float32).alias("delay_days")),
                how="cross",
            )
            .join(context.scan_internal("production_plan", ["month", "units_per_month"]), how="cross")
            .limit(1)
            .collect()
        )

        if risk_inputs.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="Production risk emerges when component inventory burn crosses replenishment lead time.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="Production-stop risk could not be computed from the available data.",
                evidence=[],
                limitations=["Missing chip inventory, production plan, or lead-time data."],
            )

        row = risk_inputs.row(0, named=True)
        days_remaining = float(row["days_remaining"])
        threshold_days = float(row["threshold_days"])
        lead_time_days = float(row["lead_time_weeks"]) * 7
        shipping_delay = float(row["delay_days"])
        days_until_threshold = max(days_remaining - threshold_days, 0.0)
        days_until_stop_risk = max(days_until_threshold - shipping_delay, 0.0)
        score = clamp(0.72 + min((lead_time_days - days_remaining) / 120.0, 0.15) + 0.03)
        confidence = 0.86
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Production risk emerges when component inventory burn crosses replenishment lead time.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Chip inventory will cross the 30-day threshold next week, and production could be at risk in 14-21 days if no action is taken.",
            evidence=[
                EvidenceItem(source="internal", label="days_until_threshold", value=days_until_threshold, detail="MCU inventory is close to the 30-day risk threshold."),
                EvidenceItem(source="external", label="lead_time_days", value=lead_time_days, detail="Replenishment lead time remains far above current coverage."),
                EvidenceItem(source="derived", label="production_risk_window_days", value=days_until_stop_risk, detail="The combined inventory and delay picture points to a near-term production-stop risk window."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Protect the next production batch",
                    detail="Switch the next batch to a backup supplier or expedite the current supplier order.",
                    urgency="critical",
                )
            ],
            limitations=["The stop-risk window is a deterministic estimate from demo data rather than a probabilistic simulation."],
            monitoring_rules=[{"type": "production_stop_risk_days", "threshold_days": 21}],
        )
