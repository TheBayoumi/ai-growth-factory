from __future__ import annotations

import unittest

from factory.models import FailedSegment
from factory.production_voice_editorial_pacing_v28 import (
    editorial_segment_factor_v28,
    editorial_segment_target_v28,
    ground_editorial_pace_failure_v28,
    segment_candidate_publishable_v28,
)
from factory.video_profile import VideoProfile


class ProductionVoiceEditorialPacingV28Tests(unittest.TestCase):
    def test_sentence_target_offsets_short_joins_without_speeding_track(self) -> None:
        self.assertEqual(editorial_segment_target_v28(VideoProfile()), 144.0)

    def test_naturally_slower_sentence_can_be_publishable(self) -> None:
        result = editorial_segment_factor_v28(116.0, profile=VideoProfile())
        self.assertIsNotNone(result)
        assert result is not None
        factor, projected = result
        self.assertEqual(factor, 1.15)
        self.assertAlmostEqual(projected, 133.4, places=1)
        self.assertTrue(segment_candidate_publishable_v28(116.0, profile=VideoProfile()))

    def test_extremely_slow_sentence_still_fails_closed(self) -> None:
        self.assertIsNone(editorial_segment_factor_v28(108.0, profile=VideoProfile()))
        self.assertFalse(segment_candidate_publishable_v28(108.0, profile=VideoProfile()))

    def test_no_sentence_correction_exceeds_115_percent(self) -> None:
        result = editorial_segment_factor_v28(121.0, profile=VideoProfile())
        self.assertIsNotNone(result)
        assert result is not None
        factor, projected = result
        self.assertLessEqual(factor, 1.15)
        self.assertGreaterEqual(projected, 132.0)

    def test_pace_only_feedback_is_cleared_inside_editorial_range(self) -> None:
        failure = FailedSegment(2, "This sentence is slightly slow.", "Speak faster.")
        self.assertIsNone(
            ground_editorial_pace_failure_v28(
                failure,
                measured_wpm=133.4,
                profile=VideoProfile(),
            )
        )

    def test_pause_feedback_keeps_sentence_pace_unchanged(self) -> None:
        failure = FailedSegment(
            2,
            "The sentence is slightly slow and the pauses are awkward.",
            "Speak faster and improve pauses.",
        )
        adjusted = ground_editorial_pace_failure_v28(
            failure,
            measured_wpm=133.4,
            profile=VideoProfile(),
        )
        self.assertIsNotNone(adjusted)
        assert adjusted is not None
        self.assertIn("Preserve this sentence near 133 words per minute", adjusted.tts_instruction)
        self.assertIn("Improve only clause-boundary pauses", adjusted.tts_instruction)
        self.assertNotIn("Speak faster", adjusted.tts_instruction)

    def test_real_pace_failure_below_editorial_range_is_retained(self) -> None:
        failure = FailedSegment(2, "The sentence drags.", "Speak faster.")
        self.assertEqual(
            ground_editorial_pace_failure_v28(
                failure,
                measured_wpm=120.0,
                profile=VideoProfile(),
            ),
            failure,
        )


if __name__ == "__main__":
    unittest.main()
