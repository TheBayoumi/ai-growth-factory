import unittest

from factory.policy import Observation, all_strategies, reward, select_strategy


class PolicyTests(unittest.TestCase):
    def test_reward_prefers_growth(self):
        weak = Observation("x", 100, 1, 0, 0, 0, 1, 20, 24)
        strong = Observation("x", 100, 10, 3, 2, 4, 0, 85, 24)
        self.assertGreater(reward(strong), reward(weak))

    def test_strategy_is_deterministic_for_seed(self):
        self.assertEqual(select_strategy([], 42), select_strategy([], 42))

    def test_strategy_space(self):
        strategies = all_strategies()
        self.assertEqual(len(strategies), 288)
        self.assertEqual(len({item.tag for item in strategies}), len(strategies))


if __name__ == "__main__":
    unittest.main()
