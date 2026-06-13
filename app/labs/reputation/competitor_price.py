from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class CompetitorPriceLab(BaseLab):
    lab_id = "competitor_price"
    lab_name = "Competitor Price Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        strongest_move = (
            context.scan_text_signals(columns=["entity_name", "label", "signal_type", "numeric_value", "evidence_text", "source_name"])
            .filter((pl.col("signal_type") == "competitor_move") & (pl.col("label") == "competitor_discount"))
            .sort("numeric_value", descending=True)
            .limit(1)
            .collect()
        )

        if strongest_move.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="A competitor price drop or new menu launch can create a local demand or revenue risk.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No competitor pricing data was available.",
                evidence=[],
                limitations=["Missing competitor menu and price records."],
            )

        changed_record = strongest_move.row(0, named=True)
        largest_drop = float(changed_record["numeric_value"])
        new_menu_count = int(
            context.scan_text_signals(columns=["label", "signal_type"])
            .filter((pl.col("signal_type") == "competitor_move") & (pl.col("label") == "new_menu"))
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        score = clamp(0.55 + largest_drop + (0.12 if new_menu_count else 0.0))
        confidence = 0.74
        status = default_status(score, confidence)
        competitor_name = str(changed_record["entity_name"])

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="A competitor price drop or new menu launch can create a local demand or revenue risk.",
            status=status,
            score=score,
            confidence=confidence,
            summary=(
                f"In the cached demo dataset, {competitor_name} reduced selected coffee prices by {round(largest_drop * 100):d}% "
                "and a nearby competitor introduced a new seasonal menu."
            ),
            evidence=[
                EvidenceItem(source="external", label="price_drop_pct", value=round(largest_drop, 2), detail=f"Cached demo data suggests {competitor_name} reduced selected prices by {round(largest_drop * 100):d}%."),
                EvidenceItem(source="external", label="affected_product", value=str(changed_record["source_name"]), detail="Affected product category is core coffee beverages."),
                EvidenceItem(source="external", label="seasonal_menu_launches", value=new_menu_count, detail="Another competitor launched a seasonal menu in cached demo data."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Review nearby pricing",
                    detail="Review pricing around affected products and monitor revenue impact in nearby branches.",
                    urgency="high",
                )
            ],
            limitations=["Competitor claims come from cached demo records and should not be presented as live-validated facts."],
            monitoring_rules=[{"type": "competitor_price_drop", "threshold": 0.10}],
        )
