from __future__ import annotations

import unittest

from app.labs.decision_cards import compile_decision_cards
from app.labs.runner import build_report, load_demo_context, run_all_labs


class DecisionCardTests(unittest.TestCase):
    def test_reputation_cards_include_operational_ensemble_and_competitor_card(self) -> None:
        context = load_demo_context("reputation_monitor")
        results = run_all_labs("reputation_monitor", context)
        report = build_report(context, results)
        cards = compile_decision_cards(report)
        card_ids = [card.card_id for card in cards]
        self.assertIn("reputation_operational_risk", card_ids)
        self.assertIn("card-competitor_price", card_ids)
        self.assertNotIn("card-location_sentiment", card_ids)

    def test_supply_chain_cards_include_production_ensemble_and_battery_card(self) -> None:
        context = load_demo_context("supply_chain_risk")
        results = run_all_labs("supply_chain_risk", context)
        report = build_report(context, results)
        cards = compile_decision_cards(report)
        card_ids = [card.card_id for card in cards]
        self.assertIn("supply_chain_production_risk", card_ids)
        self.assertIn("card-battery_cost", card_ids)
        self.assertNotIn("card-chip_supply", card_ids)


if __name__ == "__main__":
    unittest.main()
