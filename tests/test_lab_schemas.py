from __future__ import annotations

import unittest

import polars as pl

from app.labs.runner import build_report
from app.labs.schemas import EvidenceItem, LabContext, LabResult, LabRunReport, RecommendedAction


class LabSchemaTests(unittest.TestCase):
    def test_lab_result_schema_validates(self) -> None:
        result = LabResult(
            lab_id="test_lab",
            lab_name="Test Lab",
            scenario="reputation_monitor",
            hypothesis="Test hypothesis",
            status="selected",
            score=0.9,
            confidence=0.8,
            summary="A strong signal was found.",
            evidence=[EvidenceItem(source="derived", label="score", value=0.9)],
            recommended_actions=[RecommendedAction(title="Act", detail="Do the thing.")],
        )
        self.assertEqual(result.lab_id, "test_lab")
        self.assertEqual(result.evidence[0].label, "score")

    def test_context_supports_lazy_polars_scans(self) -> None:
        context = LabContext.from_records(
            scenario="reputation_monitor",
            internal_data=[{"dataset": "locations", "location_id": "one"}],
            external_data=[{"dataset": "reviews", "location_id": "one", "period": "recent"}],
        )
        frame = context.scan_internal("locations", ["location_id"])
        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(frame.collect().item(0, "location_id"), "one")

    def test_lab_run_report_groups_results(self) -> None:
        context = LabContext.from_records(scenario="reputation_monitor", internal_data=[], external_data=[])
        selected = LabResult(
            lab_id="selected_lab",
            lab_name="Selected Lab",
            scenario="reputation_monitor",
            hypothesis="H",
            status="selected",
            score=0.8,
            confidence=0.8,
            summary="Selected summary.",
            evidence=[EvidenceItem(source="derived", label="e")],
        )
        hidden = LabResult(
            lab_id="hidden_lab",
            lab_name="Hidden Lab",
            scenario="reputation_monitor",
            hypothesis="H",
            status="hidden",
            score=0.3,
            confidence=0.7,
            summary="Hidden summary.",
            evidence=[EvidenceItem(source="derived", label="e")],
        )
        report = build_report(context, [selected, hidden])
        self.assertIsInstance(report, LabRunReport)
        self.assertEqual(len(report.selected), 1)
        self.assertEqual(len(report.hidden), 1)


if __name__ == "__main__":
    unittest.main()
