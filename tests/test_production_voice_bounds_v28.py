from __future__ import annotations

import unittest

from factory.production_voice_bounds_v28 import bounded_tempo_factor_v28


class ProductionVoiceBoundsV28Tests(unittest.TestCase):
    def test_returns_exact_ceiling_when_it_can_reach_lower_gate(self) -> None:
        factor = bounded_tempo_factor_v28(
            estimated_wpm=120.0,
            target_wpm=142,
            tolerance=4,
        )
        self.assertEqual(factor, 1.15)
        self.assertLessEqual(factor, 1.15)

    def test_refuses_over_speed_when_ceiling_cannot_reach_gate(self) -> None:
        factor = bounded_tempo_factor_v28(
            estimated_wpm=110.0,
            target_wpm=142,
            tolerance=4,
        )
        self.assertIsNone(factor)

    def test_never_honors_a_legacy_145_percent_default(self) -> None:
        factor = bounded_tempo_factor_v28(
            estimated_wpm=120.0,
            target_wpm=142,
            tolerance=4,
            maximum_factor=1.45,
        )
        self.assertEqual(factor, 1.15)

    def test_no_correction_inside_the_accepted_range(self) -> None:
        self.assertIsNone(
            bounded_tempo_factor_v28(
                estimated_wpm=140.0,
                target_wpm=142,
                tolerance=4,
            )
        )


if __name__ == "__main__":
    unittest.main()
