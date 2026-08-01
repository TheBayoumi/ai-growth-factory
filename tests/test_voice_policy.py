import unittest

from factory.models import VoiceContract
from factory.policy import Strategy
from factory.voice_policy import contract_for_strategy


class VoicePolicyTests(unittest.TestCase):
    def test_fast_breaking_strategy_increases_pace_and_hook_energy(self):
        base = VoiceContract()
        contract = contract_for_strategy(
            base, Strategy("breaking", "fast", "kinetic", "55-62", "subscribe")
        )
        self.assertGreater(contract.target_wpm, base.target_wpm)
        self.assertGreater(contract.energy, base.energy)
        self.assertGreater(contract.hook_intensity, base.hook_intensity)
        self.assertIn("urgent", contract.baseline_style)

    def test_practical_balanced_strategy_remains_warm_and_bounded(self):
        contract = contract_for_strategy(
            VoiceContract(energy=0.99, warmth=0.99),
            Strategy("practical", "balanced", "dashboard", "63-72", "product"),
        )
        self.assertEqual(contract.energy, 1.0)
        self.assertEqual(contract.warmth, 1.0)
        self.assertEqual(contract.target_wpm, 155)


if __name__ == "__main__":
    unittest.main()
