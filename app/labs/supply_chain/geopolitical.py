from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class GeopoliticalLab(BaseLab):
    lab_id = "geopolitical"
    lab_name = "Geopolitical Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        news_signal = (
            context.scan_text_signals(columns=["entity_name", "signal_type", "label", "numeric_value"])
            .filter((pl.col("signal_type") == "geopolitical_risk") & (pl.col("label") == "geopolitical_risk"))
            .select(
                pl.col("entity_name").alias("region"),
                pl.col("numeric_value").cast(pl.Float32).alias("risk_score"),
            )
            .limit(1)
            .collect()
        )

        if news_signal.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="News, regulatory, or geopolitical changes can predict supplier disruption.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No geopolitical news signal was available.",
                evidence=[],
                limitations=["Missing geopolitical/news records."],
            )

        score = clamp(float(news_signal.item(0, "risk_score")))
        confidence = 0.61
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="News, regulatory, or geopolitical changes can predict supplier disruption.",
            status=status,
            score=score,
            confidence=confidence,
            summary="No strong geopolitical disruption signal was found in the cached demo data. Continue monitoring.",
            evidence=[
                EvidenceItem(source="external", label="risk_score", value=score, detail="Cached news signals do not indicate an immediate Taiwan or sanctions trigger."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Continue passive monitoring",
                    detail="No immediate action is needed beyond continued monitoring of supplier-region news.",
                    urgency="low",
                )
            ],
            limitations=["Geopolitical signals are weak in the cached demo dataset."],
            monitoring_rules=[{"type": "news_risk_score", "threshold": 0.55}],
        )
