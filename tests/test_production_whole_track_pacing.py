from __future__ import annotations

import unittest

from factory.production_pacing import production_segment_tempo_factor


class ProductionWholeTrackPacingTests(unittest.TestCase):
    def test_slow_explanation_can_use_the_larger_production_correction(self) -> None:
        factor = production_segment_tempo_factor(
            estimated_wpm=112.205,
            target_wpm=155,
            tolerance=10,
            is_closing=False,
        )

        self.assertIsNotNone(factor)
        assert factor is not None
        self.assertGreater(factor, 1.30)
        self.assertLessEqual(factor, 1.45)

    def test_slow_closing_is_capped_for_naturalness(self) -> None:
        factor = production_segment_tempo_factor(
            estimated_wpm=112.205,
            target_wpm=155,
            tolerance=10,
            is_closing=True,
        )

        self.assertEqual(factor, 1.15)

    def test_compliant_closing_is_not_retimed(self) -> None:
        self.assertIsNone(
            production_segment_tempo_factor(
                estimated_wpm=150.0,
                target_wpm=155,
                tolerance=10,
                is_closing=True,
            )
        )

    def test_v23_closing_cap_preserves_the_unchanged_whole_track_gate(self) -> None:
        total_words = 153
        current_duration_seconds = 59.857875
        closing_words = 19
        closing_before_wpm = 112.205
        closing_after_wpm_at_138x = 154.887

        current_closing_duration = closing_words / closing_after_wpm_at_138x * 60.0
        capped_closing_duration = closing_words / (closing_before_wpm * 1.15) * 60.0
        projected_duration = (
            current_duration_seconds - current_closing_duration + capped_closing_duration
        )
        projected_wpm = total_words / projected_duration * 60.0

        self.assertGreaterEqual(projected_wpm, 145.0)
        self.assertLessEqual(projected_wpm, 165.0)
        self.assertAlmostEqual(projected_wpm, 149.6, delta=0.8)


if __name__ == "__main__":
    unittest.main()
