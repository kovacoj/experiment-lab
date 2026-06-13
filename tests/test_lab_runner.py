from __future__ import annotations

import json
import unittest

from app.labs.base import BaseLab
from app.labs.runner import build_benchmark_context, build_report, list_labs, load_demo_context, render_lab_list, render_report_json, run_all_labs, run_demo_scenario, run_labs, select_top_labs
from app.labs.schemas import EvidenceItem, LabContext, LabResult


class ExplodingLab(BaseLab):
    lab_id = "exploding"
    lab_name = "Exploding Lab"
    scenario = "reputation_monitor"

    def run(self, context: LabContext) -> LabResult:
        raise RuntimeError("boom")


class StaticLab(BaseLab):
    def __init__(self, lab_id: str, score: float, status: str) -> None:
        self.lab_id = lab_id
        self.lab_name = lab_id
        self.scenario = "reputation_monitor"
        self.score = score
        self.status = status

    def run(self, context: LabContext) -> LabResult:
        return LabResult(
            lab_id=self.lab_id,
            lab_name=self.lab_name,
            scenario=context.scenario,
            hypothesis="H",
            status=self.status,
            score=self.score,
            confidence=0.8,
            summary=self.lab_id,
            evidence=[EvidenceItem(source="derived", label="e")],
        )


class LabRunnerTests(unittest.TestCase):
    def test_runner_catches_lab_exceptions(self) -> None:
        context = load_demo_context("reputation_monitor")
        results = run_labs(context, [ExplodingLab()])
        self.assertEqual(results[0].status, "failed")
        self.assertIn("boom", results[0].summary)

    def test_select_top_labs_caps_selected_and_warning(self) -> None:
        context = load_demo_context("reputation_monitor")
        labs = [
            StaticLab("a", 0.95, "selected"),
            StaticLab("b", 0.90, "selected"),
            StaticLab("c", 0.85, "selected"),
            StaticLab("d", 0.80, "selected"),
            StaticLab("e", 0.60, "warning"),
            StaticLab("f", 0.55, "warning"),
            StaticLab("g", 0.50, "warning"),
        ]
        results = run_labs(context, labs)
        grouped = select_top_labs(results)
        self.assertEqual(len(grouped["selected"]), 3)
        self.assertEqual(len(grouped["warning"]), 2)
        self.assertTrue(any(item.lab_id == "d" for item in grouped["hidden"]))
        self.assertTrue(any(item.lab_id == "g" for item in grouped["hidden"]))

    def test_render_report_json_returns_valid_payload(self) -> None:
        context = load_demo_context("reputation_monitor")
        results = run_all_labs("reputation_monitor", context)
        report = build_report(context, results)
        payload = json.loads(render_report_json(report, context, runtime_seconds=0.1234))
        self.assertEqual(payload["scenario"], "reputation_monitor")
        self.assertIn("selected", payload)
        self.assertIn("runtime_seconds", payload)
        self.assertIn("ensembles", payload)
        self.assertIn("discarded", payload)

    def test_build_benchmark_context_expands_external_rows(self) -> None:
        context = build_benchmark_context("reputation_monitor", 40)
        self.assertEqual(context.metadata["benchmark_rows"], 40)
        self.assertEqual(context.external_data.count_rows(), 40)

    def test_list_labs_returns_known_registry_entries(self) -> None:
        labs = list_labs("reputation_monitor")
        lab_ids = [lab["lab_id"] for lab in labs]
        self.assertIn("location_sentiment", lab_ids)
        self.assertIn("competitor_price", lab_ids)

    def test_run_demo_scenario_can_run_single_lab(self) -> None:
        _, report = run_demo_scenario("reputation_monitor", lab_ids=["location_sentiment"])
        selected_ids = [result.lab_id for result in report.selected]
        all_ids = selected_ids + [result.lab_id for result in report.warning] + [result.lab_id for result in report.hidden] + [result.lab_id for result in report.discarded]
        self.assertEqual(all_ids, ["location_sentiment"])

    def test_render_lab_list_is_readable(self) -> None:
        output = render_lab_list("reputation_monitor", list_labs("reputation_monitor"))
        self.assertIn("Available labs", output)
        self.assertIn("location_sentiment", output)


if __name__ == "__main__":
    unittest.main()
