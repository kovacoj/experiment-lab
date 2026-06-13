from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import clamp, default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class AlternativeSupplierLab(BaseLab):
    lab_id = "alternative_supplier"
    lab_name = "Alternative Supplier Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        alternatives = (
            context.scan_text_signals(columns=["entity_name", "signal_type", "label", "numeric_value", "confidence"])
            .filter((pl.col("signal_type") == "supplier_option") & (pl.col("label") == "alternative_supplier_available"))
            .with_columns(
                pl.col("numeric_value").cast(pl.Float32),
                pl.col("confidence").cast(pl.Float32),
            )
            .collect()
        )

        if alternatives.is_empty():
            return LabResult(
                lab_id=self.lab_id,
                lab_name=self.lab_name,
                scenario=context.scenario,
                hypothesis="A higher-cost backup supplier may be justified if production-stop risk is high.",
                status="inconclusive",
                score=0.0,
                confidence=0.0,
                summary="No approved backup suppliers were found.",
                evidence=[],
                limitations=["Missing approved backup supplier records."],
            )

        backup_supplier_count = alternatives.height
        price_premium = float(alternatives.select(pl.max("numeric_value")).item(0, 0)) / 100.0
        compatibility_confidence = float(alternatives.select(pl.min("confidence")).item(0, 0))
        score = clamp(0.30 + min(backup_supplier_count / 10.0, 0.20) + 0.02)
        confidence = clamp(compatibility_confidence - 0.25)
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="A higher-cost backup supplier may be justified if production-stop risk is high.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Two backup MCU suppliers are available in the cached demo dataset, but they are materially more expensive and should stay contingency-only for now.",
            evidence=[
                EvidenceItem(source="external", label="backup_supplier_count", value=backup_supplier_count, detail="Two approved backup suppliers are on file in cached supplier text data."),
                EvidenceItem(source="derived", label="price_premium_pct", value=price_premium, detail="The most expensive backup option is about 40% above the current supplier."),
                EvidenceItem(source="derived", label="compatibility_confidence", value=compatibility_confidence, detail="Compatibility looks promising but is not enough to justify an automatic switch."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Prepare contingency approval",
                    detail="Prepare backup supplier approval, but do not switch unless chip risk remains high.",
                    urgency="medium",
                )
            ],
            limitations=["Alternative supplier evidence is useful for contingency planning, but the cost premium is substantial."],
            monitoring_rules=[{"type": "backup_supplier_ready", "component": "MCU chip"}],
        )
