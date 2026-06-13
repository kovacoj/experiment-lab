from __future__ import annotations

import polars as pl

from app.labs.base import BaseLab
from app.labs.helpers import default_status
from app.labs.schemas import EvidenceItem, LabContext, LabResult, RecommendedAction


class SupplyChainDataQualityLab(BaseLab):
    lab_id = "supply_chain_data_quality"
    lab_name = "Data Quality Lab"
    scenario = "supply_chain_risk"

    def run(self, context: LabContext) -> LabResult:
        inventory_count = int(context.scan_internal("inventory", ["component"]).select(pl.len()).collect().item(0, 0))
        production_count = int(context.scan_internal("production_plan", ["month"]).select(pl.len()).collect().item(0, 0))
        supplier_count = int(context.scan_internal("suppliers", ["supplier_name"]).select(pl.len()).collect().item(0, 0))
        external_signal_count = int(context.scan_text_signals(columns=["document_id"]).select(pl.len()).collect().item(0, 0))

        checks = [inventory_count > 0, production_count > 0, supplier_count > 0, external_signal_count >= 4]
        score = 0.58 if all(checks) else 0.52
        confidence = 0.60
        status = default_status(score, confidence)

        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="Supply-risk analysis needs inventory, production, supplier, and external signal coverage.",
            status=status,
            score=score,
            confidence=confidence,
            summary="Inventory, production, and supplier data are usable. External commodity and shipping data is cached and should be treated as indicative.",
            evidence=[
                EvidenceItem(source="internal", label="inventory_components", value=inventory_count, detail="Critical components are present in inventory records."),
                EvidenceItem(source="internal", label="supplier_records", value=supplier_count, detail="Primary and backup supplier records are available."),
                EvidenceItem(source="external", label="external_signal_count", value=external_signal_count, detail="Cached lead-time, price, shipping, and news signals are available."),
            ],
            recommended_actions=[
                RecommendedAction(
                    title="Treat cached external feeds as directional",
                    detail="Validate critical commodity or logistics shifts with live feeds before issuing production decisions.",
                    urgency="medium",
                )
            ],
            limitations=["External commodity and shipping data is cached demo data rather than live market infrastructure."],
            monitoring_rules=[{"type": "minimum_component_coverage", "minimum_components": 4}],
        )
