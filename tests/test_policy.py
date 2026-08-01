from __future__ import annotations

import unittest

from factory.policy import Observation, Strategy, reward, select_strategy


class PolicyTests(unittest.TestCase):
    def test_strategy_tag_is_stable(self) -> None:
        strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")
        self.assertEqual(strategy.tag, strategy.tag)
        self.assertTrue(strategy.tag.startswith("agfs-"))

    def test_subscriber_growth_increases_reward(self) -> None:
        base = Observation("x", 1000, 30, 5, 2, 0, 0, 60, 24)
        growth = Observation("x", 1000, 30, 5, 2, 12, 0, 60, 24)
        self.assertGreater(reward(growth), reward(base))

    def test_cold_start_returns_strategy(self) -> None:
        self.assertIsInstance(select_strategy([], 42), Strategy)


if __name__ == "__main__":
    unittest.main()
