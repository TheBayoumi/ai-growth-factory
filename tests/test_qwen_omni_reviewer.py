import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.models import AudioMetrics, NarrationSegment, VoiceContract
from factory.qwen_omni_reviewer import QwenOmniReviewer
from factory.reviewer import ReviewerError


PASSING_SCORES = {
    "script_fidelity": 0.99,
    "naturalness": 0.91,
    "authority": 0.90,
    "engagement": 0.89,
    "pronunciation": 0.96,
    "pace": 0.90,
    "pause_quality": 0.89,
    "emotional_match": 0.88,
    "audio_artifacts": 0.97,
}


def metrics() -> AudioMetrics:
    return AudioMetrics(
        duration_seconds=8.0,
        sample_rate=24000,
        channels=1,
        peak_dbfs=-1.5,
        rms_dbfs=-18.0,
        clipping_ratio=0.0,
        silence_ratio=0.08,
        max_silence_seconds=0.4,
        estimated_wpm=154.0,
        dc_offset=0.0,
        passed=True,
    )


class QwenOmniReviewerTests(unittest.TestCase):
    def _segments(self, root: Path) -> list[NarrationSegment]:
        return [
            NarrationSegment(
                segment_id=index,
                text=f"Segment {index} exact narration text.",
                instruction="Speak clearly.",
                audio_path=root / f"segment-{index}.wav",
                start_seconds=index * 4.0,
                end_seconds=(index + 1) * 4.0,
            )
            for index in range(2)
        ]

    def test_aggregates_approved_segment_reviews(self):
        def infer(segment, contract, audio_metrics, attempt):
            del segment, contract, audio_metrics, attempt
            return {
                "decision": "approve",
                "overall_score": 0.92,
                "scores": PASSING_SCORES,
                "reason": "",
                "tts_instruction": "",
            }

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ", {}, clear=True
        ):
            settings = Settings.from_env()
            reviewer = QwenOmniReviewer(settings, segment_inference=infer)
            review = reviewer.review(
                audio_path=Path(temporary) / "voice.wav",
                narration="Ignored because segments carry the exact transcript.",
                contract=VoiceContract(),
                segments=self._segments(Path(temporary)),
                metrics=metrics(),
                attempt=1,
            )
        self.assertEqual(review.decision, "approve")
        self.assertEqual(review.failed_segments, ())
        self.assertAlmostEqual(review.overall_score, 0.92)
        self.assertEqual(review.reviewer_model, settings.qwen_omni_model)

    def test_maps_retry_to_exact_segment(self):
        def infer(segment, contract, audio_metrics, attempt):
            del contract, audio_metrics, attempt
            if segment.segment_id == 1:
                return {
                    "decision": "retry",
                    "overall_score": 0.78,
                    "scores": {**PASSING_SCORES, "naturalness": 0.70},
                    "reason": "Robotic cadence in the second sentence.",
                    "tts_instruction": "Use smoother phrasing and a shorter mid-sentence pause.",
                }
            return {
                "decision": "approve",
                "overall_score": 0.93,
                "scores": PASSING_SCORES,
                "reason": "",
                "tts_instruction": "",
            }

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ", {}, clear=True
        ):
            settings = Settings.from_env()
            review = QwenOmniReviewer(settings, segment_inference=infer).review(
                audio_path=Path(temporary) / "voice.wav",
                narration="Narration",
                contract=VoiceContract(),
                segments=self._segments(Path(temporary)),
                metrics=metrics(),
                attempt=1,
            )
        self.assertEqual(review.decision, "retry_segments")
        self.assertEqual([item.segment_id for item in review.failed_segments], [1])
        self.assertIn("smoother phrasing", review.failed_segments[0].tts_instruction)

    def test_retry_requires_executable_instruction(self):
        def infer(segment, contract, audio_metrics, attempt):
            del segment, contract, audio_metrics, attempt
            return {
                "decision": "retry",
                "overall_score": 0.7,
                "scores": PASSING_SCORES,
                "reason": "Flat delivery.",
                "tts_instruction": "",
            }

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ", {}, clear=True
        ):
            settings = Settings.from_env()
            with self.assertRaises(ReviewerError):
                QwenOmniReviewer(settings, segment_inference=infer).review(
                    audio_path=Path(temporary) / "voice.wav",
                    narration="Narration",
                    contract=VoiceContract(),
                    segments=self._segments(Path(temporary)),
                    metrics=metrics(),
                    attempt=1,
                )


if __name__ == "__main__":
    unittest.main()
