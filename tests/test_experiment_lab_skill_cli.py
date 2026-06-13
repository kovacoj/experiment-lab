from __future__ import annotations

import json
import subprocess
import sys
import unittest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "app.agent_skills.experiment_lab_cli", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd="/home/cady/personal/orchestrator/experiment-lab",
    )


class ExperimentLabSkillCliTests(unittest.TestCase):
    def test_list_scenarios_returns_expected_values(self) -> None:
        result = run_cli("list-scenarios")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertIn("reputation_monitor", payload["scenarios"])
        self.assertIn("supply_chain_risk", payload["scenarios"])

    def test_list_labs_works_for_reputation(self) -> None:
        result = run_cli("list-labs", "--scenario", "reputation_monitor")
        payload = json.loads(result.stdout)
        lab_ids = [lab["lab_id"] for lab in payload["labs"]]
        self.assertEqual(result.returncode, 0)
        self.assertIn("location_sentiment", lab_ids)

    def test_list_labs_works_for_supply_chain(self) -> None:
        result = run_cli("list-labs", "--scenario", "supply_chain_risk")
        payload = json.loads(result.stdout)
        lab_ids = [lab["lab_id"] for lab in payload["labs"]]
        self.assertEqual(result.returncode, 0)
        self.assertIn("chip_supply", lab_ids)

    def test_run_analysis_reputation_returns_ok_json(self) -> None:
        result = run_cli("run-analysis", "--scenario", "reputation_monitor", "--format", "json")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["scenario"], "reputation_monitor")
        self.assertIn("decision_cards", payload)

    def test_run_analysis_supply_chain_returns_ok_json(self) -> None:
        result = run_cli("run-analysis", "--scenario", "supply_chain_risk", "--format", "json")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["scenario"], "supply_chain_risk")

    def test_refresh_compare_returns_prediction_changed_field(self) -> None:
        result = run_cli("refresh-compare", "--scenario", "reputation_monitor", "--format", "json")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertIn("prediction_changed", payload)

    def test_invalid_scenario_returns_error_json(self) -> None:
        result = run_cli("run-analysis", "--scenario", "invalid_scenario", "--format", "json")
        payload = json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "error")

    def test_outputs_parse_as_json(self) -> None:
        for args in [
            ("list-scenarios",),
            ("list-labs", "--scenario", "reputation_monitor"),
            ("run-analysis", "--scenario", "reputation_monitor", "--format", "json"),
            ("refresh-compare", "--scenario", "supply_chain_risk", "--format", "json"),
        ]:
            result = run_cli(*args)
            json.loads(result.stdout)


if __name__ == "__main__":
    unittest.main()
