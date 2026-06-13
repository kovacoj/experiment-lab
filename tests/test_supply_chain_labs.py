from __future__ import annotations

import unittest

from app.labs.runner import build_report, load_demo_context, run_all_labs


class SupplyChainLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = load_demo_context("supply_chain_risk")
        cls.results = run_all_labs("supply_chain_risk", cls.context)
        cls.by_id = {result.lab_id: result for result in cls.results}

    def test_each_lab_returns_lab_result(self) -> None:
        self.assertEqual(len(self.results), 7)

    def test_chip_supply_lab_matches_expected_signal(self) -> None:
        result = self.by_id["chip_supply"]
        self.assertGreater(result.score, 0.85)
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("16-week" in (item.detail or "") or "30-day" in (item.detail or "") for item in result.evidence))

    def test_battery_cost_lab_matches_expected_signal(self) -> None:
        result = self.by_id["battery_cost"]
        self.assertGreater(result.score, 0.80)
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("34%" in (item.detail or "") or "15%" in (item.detail or "") for item in result.evidence))

    def test_shipping_risk_lab_matches_expected_signal(self) -> None:
        result = self.by_id["shipping_risk"]
        self.assertGreater(result.score, 0.75)
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("5-day" in (item.detail or "") or item.value == 5 for item in result.evidence))

    def test_geopolitical_lab_stays_hidden(self) -> None:
        result = self.by_id["geopolitical"]
        self.assertEqual(result.status, "hidden")
        self.assertLess(result.score, 0.50)

    def test_production_stop_risk_lab_is_selected(self) -> None:
        result = self.by_id["production_stop_risk"]
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("threshold" in (item.detail or "").lower() or "risk" in (item.detail or "").lower() for item in result.evidence))

    def test_report_selects_expected_top_labs(self) -> None:
        report = build_report(self.context, self.results)
        selected_ids = [result.lab_id for result in report.selected]
        self.assertEqual(selected_ids, ["chip_supply", "production_stop_risk", "battery_cost"])

    def test_supply_chain_production_risk_ensemble_combines_key_labs(self) -> None:
        report = build_report(self.context, self.results)
        ensemble = next(item for item in report.ensembles if item.ensemble_id == "supply_chain_production_risk")
        self.assertIn("chip_supply", ensemble.contributing_lab_ids)
        self.assertIn("shipping_risk", ensemble.contributing_lab_ids)
        self.assertIn("production_stop_risk", ensemble.contributing_lab_ids)

    def test_battery_cost_remains_separate_from_production_ensemble(self) -> None:
        report = build_report(self.context, self.results)
        contributing_ids = {lab_id for ensemble in report.ensembles for lab_id in ensemble.contributing_lab_ids}
        self.assertNotIn("battery_cost", contributing_ids)

    def test_geopolitical_weak_signal_remains_hidden(self) -> None:
        report = build_report(self.context, self.results)
        hidden_ids = [result.lab_id for result in report.hidden]
        self.assertIn("geopolitical", hidden_ids)


if __name__ == "__main__":
    unittest.main()
