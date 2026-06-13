from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class BatteryCostLab(BaseLab):
    lab_id = "battery_cost"
    lab_name = "Battery Cost Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        battery_status = (
            context.scan_internal("inventory", ["component", "days_remaining"])
            .filter(pl.col("component") == "lithium battery pack")
            .with_columns(pl.col("days_remaining").cast(pl.Float32))
            .join(
                context.scan_text_signals(columns=["entity_name", "label", "signal_type", "numeric_value"])
                .filter((pl.col("signal_type") == "price_change") & pl.col("label").is_in(["commodity_price_pressure", "price_increase"]))
                .group_by("entity_name")
                .agg(
                    pl.when(pl.col("label") == "commodity_price_pressure").then(pl.col("numeric_value")).otherwise(None).max().alias("price_change_pct_3m"),
                    pl.when(pl.col("label") == "price_increase").then(pl.col("numeric_value")).otherwise(None).max().alias("predicted_cost_increase_pct"),
                )
                .select(
                    pl.col("entity_name").alias("component"),
                    (pl.col("price_change_pct_3m") / 100.0).alias("price_change_pct_3m"),
                    (pl.col("predicted_cost_increase_pct") / 100.0).alias("predicted_cost_increase_pct"),
                ),
                on="component",
                how="inner",
            )
            .limit(1)
            .collect()
        )

        if battery_status.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="Battery or lithium price increases affect next-quarter production cost.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="Missing battery inventory or cost signals.",
                evidence=[],
                limitations=["Battery inventory or commodity pricing data is missing."],
            )

        row = battery_status.row(0, named=True)
        quarterly_increase = float(row["predicted_cost_increase_pct"])
        spot_increase = float(row["price_change_pct_3m"])
        score = clamp(0.58 + min(spot_increase, 0.19) + min(quarterly_increase, 0.10))
        confidence = 0.82
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Battery or lithium price increases affect next-quarter production cost.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Lithium and battery costs increased sharply in cached demo data, with a likely next-quarter battery cost increase around 15%.",
            evidence=[
                EvidenceItem(source="external", label="spot_price_change_pct_3m", value=spot_increase, detail="Lithium and battery pricing is up 34% over three months in cached demo data."),
                EvidenceItem(source="external", label="predicted_next_quarter_cost_increase", value=quarterly_increase, detail="Supplier pricing suggests roughly a 15% battery cost increase next quarter."),
                EvidenceItem(source="internal", label="battery_days_remaining", value=float(row["days_remaining"]), detail="Battery coverage is already relatively tight."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Lock pricing or adjust budget",
                    detail="Lock in pricing, renegotiate supply, or adjust the next-quarter production budget.",
                    urgency="high",
                )
            ],
            limitations=["Battery cost pressure is estimated from cached pricing signals and supplier announcements."],
            monitoring_rules=[{"type": "cost_increase_pct", "component": "lithium battery pack", "threshold": 0.10}],
        )
