from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from factory.models import FailedSegment
from factory.production_voice_capacity_v29 import (
    split_narration_for_voice_v29,
    voice_segment_capacity_v29,
)
from factory.production_voice_convergence_v28 import (
    best_reachable_calibration_event_v28,
    ground_pace_failure_v28,
    split_narration_for_voice_v28,
)
from factory.video_profile import VideoProfile


class ProductionVoiceConvergenceV28Tests(unittest.TestCase):
    def test_sentence_aligned_split_bounds_the_failed_canary_segments(self) -> None:
        narration = (
            "Microsoft Research has launched Orchard, an open-source framework designed to "
            "streamline the development of AI agents across diverse tasks. The framework "
            "enables researchers to reuse infrastructure, reducing complexity while maintaining "
            "strong performance from smaller models. This innovation accelerates progress in "
            "areas like natural language processing and robotics by simplifying the training "
            "and evaluation of AI systems. By making the framework accessible to the research "
            "community, Microsoft Research is fostering collaboration and innovation in AI. "
            "The framework is available for free use, empowering researchers to experiment with "
            "new ideas and push the boundaries of agentic AI. Sources: Microsoft Research blog "
            "post. Before adoption, read the linked source and test the claim on a controlled "
            "task. Track latency, failure rate, human corrections, and repeatability so the "
            "decision follows measured behavior rather than a polished announcement. The "
            "evidence should determine the next step, not the headline alone."
        )
        segments = split_narration_for_voice_v28(narration, 6)
        self.assertEqual(len(segments), 8)
        self.assertTrue(all(8 <= len(segment.split()) <= 24 for segment in segments))
        self.assertEqual(" ".join(segments), narration)

    def test_v29_capacity_accepts_thirteen_valid_sentence_segments(self) -> None:
        sentences = [
            "Adult learners test modular AI hardware together today.",
            "Researchers connect reusable compute modules during practical workshops.",
            "Developers calibrate camera sensors beside compact robotic mechanisms.",
            "Mentors guide participants through measurable automation experiments safely.",
            "Teams inspect physical prototypes using shared laboratory equipment.",
            "Engineers validate agent workflows across distinct test stations.",
            "Participants assemble programmable controllers with realistic sensor kits.",
            "Instructors demonstrate practical AI applications using industrial hardware.",
            "Learners compare repeatable results across controlled physical tasks.",
            "Researchers document latency failures corrections and repeatability carefully.",
            "Developers observe autonomous equipment completing one clear workflow.",
            "Teams reuse infrastructure while preserving measurable system performance.",
            "Participants finish by testing the complete prototype independently.",
        ]
        self.assertTrue(all(len(sentence.split()) == 8 for sentence in sentences))
        narration = " ".join(sentences)
        segments = split_narration_for_voice_v29(narration, 6)
        self.assertEqual(len(segments), 13)
        self.assertEqual(" ".join(segments), narration)
        self.assertTrue(all(8 <= len(segment.split()) <= 24 for segment in segments))

    def test_v29_capacity_is_derived_from_transcript_size(self) -> None:
        self.assertEqual(
            voice_segment_capacity_v29(
                total_words=104,
                target_segments=6,
                minimum_words=8,
            ),
            13,
        )
        self.assertEqual(
            voice_segment_capacity_v29(
                total_words=144,
                target_segments=6,
                minimum_words=8,
            ),
            18,
        )

    def test_v29_absolute_ceiling_remains_fail_closed(self) -> None:
        with patch.dict(os.environ, {"V29_ABSOLUTE_MAX_VOICE_SEGMENTS": "12"}):
            self.assertEqual(
                voice_segment_capacity_v29(
                    total_words=104,
                    target_segments=6,
                    minimum_words=8,
                ),
                12,
            )
            narration = " ".join(
                f"Adult learners test modular AI hardware safely round {index}."
                for index in range(13)
            )
            with self.assertRaisesRegex(ValueError, "above adaptive capacity 12"):
                split_narration_for_voice_v29(narration, 6)

    def test_best_reachable_candidate_ignores_newer_unreachable_take(self) -> None:
        events = [
            {
                "internal_attempt": 1,
                "reachable": True,
                "projected_wpm": 142.0,
                "required_tempo_factor": 1.145965,
            },
            {
                "internal_attempt": 2,
                "reachable": True,
                "projected_wpm": 142.0,
                "required_tempo_factor": 1.071228,
            },
            {
                "internal_attempt": 3,
                "reachable": False,
                "projected_wpm": 122.983,
                "required_tempo_factor": 1.15,
            },
        ]
        selected = best_reachable_calibration_event_v28(events)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["internal_attempt"], 2)

    def test_unreachable_only_candidates_have_no_selection(self) -> None:
        self.assertIsNone(
            best_reachable_calibration_event_v28(
                [
                    {
                        "internal_attempt": 1,
                        "reachable": False,
                        "projected_wpm": 106.4,
                        "required_tempo_factor": 1.15,
                    },
                    {
                        "internal_attempt": 2,
                        "reachable": False,
                        "projected_wpm": 123.0,
                        "required_tempo_factor": 1.15,
                    },
                ]
            )
        )

    def test_objective_pace_removes_pace_only_retry(self) -> None:
        failure = FailedSegment(
            segment_id=4,
            reason="The pace should be slightly faster.",
            tts_instruction="Speak faster.",
        )
        self.assertIsNone(
            ground_pace_failure_v28(
                failure,
                measured_wpm=142.2,
                profile=VideoProfile(),
            )
        )

    def test_pause_retry_preserves_valid_measured_pace(self) -> None:
        failure = FailedSegment(
            segment_id=4,
            reason="The pace could be faster and the pauses more natural.",
            tts_instruction="Speak faster and improve pauses.",
        )
        adjusted = ground_pace_failure_v28(
            failure,
            measured_wpm=142.2,
            profile=VideoProfile(),
        )
        self.assertIsNotNone(adjusted)
        assert adjusted is not None
        self.assertIn("Keep the speaking pace unchanged", adjusted.tts_instruction)
        self.assertIn("Improve only clause-boundary pauses", adjusted.tts_instruction)
        self.assertNotIn("Speak faster", adjusted.tts_instruction)

    def test_real_slow_segment_still_requires_repair(self) -> None:
        failure = FailedSegment(
            segment_id=4,
            reason="The pace is too slow.",
            tts_instruction="Speak faster.",
        )
        adjusted = ground_pace_failure_v28(
            failure,
            measured_wpm=119.0,
            profile=VideoProfile(),
        )
        self.assertEqual(adjusted, failure)


if __name__ == "__main__":
    unittest.main()
