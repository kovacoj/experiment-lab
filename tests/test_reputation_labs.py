from __future__ import annotations

import unittest

from app.labs.runner import build_report, load_demo_context, run_all_labs


class ReputationLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = load_demo_context("reputation_monitor")
        cls.results = run_all_labs("reputation_monitor", cls.context)
        cls.by_id = {result.lab_id: result for result in cls.results}

    def test_each_lab_returns_lab_result(self) -> None:
        self.assertEqual(len(self.results), 6)

    def test_location_sentiment_lab_matches_expected_signal(self) -> None:
        result = self.by_id["location_sentiment"]
        self.assertGreater(result.score, 0.80)
        self.assertEqual(result.status, "selected")
        self.assertIn("Vinohrady", result.summary)
        self.assertTrue(any("slow service" in (item.detail or "") for item in result.evidence))

    def test_competitor_price_lab_matches_expected_signal(self) -> None:
        result = self.by_id["competitor_price"]
        self.assertGreater(result.score, 0.75)
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("15" in (item.detail or "") for item in result.evidence))

    def test_peak_hours_lab_matches_expected_signal(self) -> None:
        result = self.by_id["peak_hours"]
        self.assertGreater(result.score, 0.70)
        self.assertEqual(result.status, "selected")
        self.assertTrue(any("morning" in (item.detail or "").lower() or "8-9" in str(item.value) for item in result.evidence))

    def test_menu_trend_lab_stays_weak(self) -> None:
        result = self.by_id["menu_trend"]
        self.assertIn(result.status, {"hidden", "warning"})
        self.assertLess(result.score, 0.60)

    def test_data_quality_lab_is_warning_not_failure(self) -> None:
        result = self.by_id["reputation_data_quality"]
        self.assertEqual(result.status, "warning")
        self.assertGreater(result.score, 0.50)

    def test_staff_mention_lab_stays_privacy_safe(self) -> None:
        result = self.by_id["staff_mention"]
        self.assertIn(result.status, {"hidden", "warning"})
        self.assertTrue(any("person-level evidence" in item.lower() for item in result.limitations))

    def test_report_selects_expected_top_labs(self) -> None:
        report = build_report(self.context, self.results)
        selected_ids = [result.lab_id for result in report.selected]
        self.assertEqual(selected_ids, ["location_sentiment", "competitor_price", "peak_hours"])

    def test_reputation_operational_ensemble_combines_operational_labs(self) -> None:
        report = build_report(self.context, self.results)
        ensemble = next(item for item in report.ensembles if item.ensemble_id == "reputation_operational_risk")
        self.assertIn("location_sentiment", ensemble.contributing_lab_ids)
        self.assertIn("peak_hours", ensemble.contributing_lab_ids)

    def test_competitor_price_remains_separate_from_ensemble(self) -> None:
        report = build_report(self.context, self.results)
        contributing_ids = {lab_id for ensemble in report.ensembles for lab_id in ensemble.contributing_lab_ids}
        self.assertNotIn("competitor_price", contributing_ids)

    def test_hidden_labs_remain_visible_for_transparency(self) -> None:
        report = build_report(self.context, self.results)
        hidden_by_id = {result.lab_id: result for result in report.hidden}
        self.assertIn("menu_trend", hidden_by_id)
        self.assertIn("staff_mention", hidden_by_id)
        self.assertIsNotNone(hidden_by_id["menu_trend"].reason_for_hiding)


if __name__ == "__main__":
    unittest.main()
