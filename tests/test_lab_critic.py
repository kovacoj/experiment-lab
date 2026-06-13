from __future__ import annotations

import unittest

from app.labs.critic import validate_lab_result
from app.labs.schemas import EvidenceItem, LabContext, LabResult


class LabCriticTests(unittest.TestCase):
    def test_critic_downgrades_unsupported_high_confidence_claims(self) -> None:
        context = LabContext.from_records(scenario="reputation_monitor", internal_data=[], external_data=[])
        result = LabResult(
            lab_id="unsafe_staff",
            lab_name="Unsafe Staff",
            scenario="reputation_monitor",
            hypothesis="H",
            status="selected",
            score=0.90,
            confidence=0.45,
            summary="Martina caused the issue.",
            evidence=[EvidenceItem(source="derived", label="e", detail="Martina caused the issue.")],
        )
        validated = validate_lab_result(result, context)
        self.assertEqual(validated.status, "discarded")
        self.assertIn("person-level attribution", validated.summary)
        self.assertTrue(any("privacy" in item.lower() or "causality" in item.lower() for item in validated.limitations))



if __name__ == "__main__":
    unittest.main()
