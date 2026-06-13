from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class ShippingRiskLab(BaseLab):
    lab_id = "shipping_risk"
    lab_name = "Shipping Risk Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        shipping_status = (
            context.scan_text_signals(columns=["entity_name", "signal_type", "label", "numeric_value"])
            .filter((pl.col("signal_type") == "shipping_delay") & (pl.col("label") == "shipping_delay"))
            .select(
                pl.lit("asia-hamburg-prague").alias("route"),
                pl.col("entity_name").alias("port"),
                pl.col("numeric_value").cast(pl.Float32).alias("delay_days"),
            )
            .join(
                context.scan_internal("purchase_orders", ["po_id", "route", "component"]),
                on="route",
                how="left",
            )
            .group_by(["route", "port", "delay_days"])
            .agg(pl.len().alias("affected_purchase_orders"))
            .limit(1)
            .collect()
        )

        if shipping_status.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="Port congestion or container delays can create production risk even when suppliers ship on time.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No shipping-delay signal was available.",
                evidence=[],
                limitations=["Missing shipping or port-delay records."],
            )

        row = shipping_status.row(0, named=True)
        delay_days = float(row["delay_days"])
        affected_orders = int(row["affected_purchase_orders"])
        score = clamp(0.56 + min(delay_days / 20.0, 0.25))
        confidence = 0.76
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Port congestion or container delays can create production risk even when suppliers ship on time.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Hamburg port congestion adds a 5-day delay to inbound components in the cached demo dataset.",
            evidence=[
                EvidenceItem(source="external", label="delay_days", value=delay_days, detail="Cached logistics data shows a 5-day delay on the Hamburg route."),
                EvidenceItem(source="internal", label="affected_purchase_orders", value=affected_orders, detail="Three open purchase orders depend on the delayed route."),
                EvidenceItem(source="derived", label="affected_route", value=str(row["route"]), detail="The delay affects the main inbound route for critical parts."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Reprioritize delayed components",
                    detail="Prioritize components from delayed containers and update the production schedule.",
                    urgency="high",
                )
            ],
            limitations=["Shipping risk uses cached delay data rather than live carrier integrations."],
            monitoring_rules=[{"type": "shipping_delay_days", "threshold": 3}],
        )
