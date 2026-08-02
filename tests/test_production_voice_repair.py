import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.models import AudioReview
from factory.production_voice_repair import promote_repairable_reject
from factory.voice_pipeline import build_reviewed_narration


class ToneTTS:
    def __init__(self, target_wpm: int) -> None:
        self.target_wpm = target_wpm
        self.calls: list[tuple[str, str, int]] = []

    def generate(self, *, text: str, instruction: str, output_path: Path, seed: int) -> Path:
        self.calls.append((text, instruction, seed))
        duration = max(0.7, len(text.split()) / self.target_wpm * 60)
        sample_rate = 24000
        samples = array(
            "h",
            (
                int(6500 * math.sin(2 * math.pi * 190 * index / sample_rate))
                for index in range(int(duration * sample_rate))
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())
        return output_path


class RejectThenApproveReviewer:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _scores(script_fidelity: float = 0.99) -> dict[str, float]:
        return {
            "script_fidelity": script_fidelity,
            "naturalness": 0.92,
            "authority": 0.92,
            "engagement": 0.91,
            "pronunciation": 0.97,
            "pace": 0.92,
            "pause_quality": 0.92,
            "emotional_match": 0.91,
            "audio_artifacts": 0.97,
        }

    def review(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls == 1:
            raw = AudioReview.from_dict(
                {
                    "decision": "reject",
                    "overall_score": 0.84,
                    "scores": self._scores(script_fidelity=0.70),
                    "failed_segments": [
                        {
                            "segment_id": 1,
                            "reason": "The exact supplied words were not spoken.",
                            "tts_instruction": "Regenerate this segment with exact script fidelity and clean audio.",
                        }
                    ],
                    "summary": "One segment needs an exact-transcript repair.",
                },
                model="qwen-omni-test",
            )
            return promote_repairable_reject(raw)
        return AudioReview.from_dict(
            {
                "decision": "approve",
                "overall_score": 0.95,
                "scores": self._scores(),
                "failed_segments": [],
                "summary": "Approved.",
            },
            model="qwen-omni-test",
        )


class ProductionVoiceRepairTests(unittest.TestCase):
    def test_actionable_reject_is_promoted_to_segment_retry(self):
        review = AudioReview.from_dict(
            {
                "decision": "reject",
                "overall_score": 0.5,
                "scores": RejectThenApproveReviewer._scores(script_fidelity=0.4),
                "failed_segments": [
                    {
                        "segment_id": 0,
                        "reason": "Words were omitted.",
                        "tts_instruction": "Speak every supplied word exactly.",
                    }
                ],
                "summary": "Repairable transcript mismatch.",
            },
            model="qwen-omni-test",
        )

        promoted = promote_repairable_reject(review)

        self.assertEqual(promoted.decision, "retry_segments")
        self.assertEqual(promoted.failed_segments, review.failed_segments)
        self.assertEqual(promoted.overall_score, review.overall_score)

    def test_reject_without_actionable_feedback_remains_hard_reject(self):
        review = AudioReview.from_dict(
            {
                "decision": "reject",
                "overall_score": 0.2,
                "scores": RejectThenApproveReviewer._scores(script_fidelity=0.2),
                "failed_segments": [],
                "summary": "Severe corruption without a safe repair instruction.",
            },
            model="qwen-omni-test",
        )

        self.assertIs(promote_repairable_reject(review), review)

    def test_repairable_reject_regenerates_only_named_segment(self):
        narration = (
            "The first segment introduces a concrete artificial intelligence update. "
            "The second segment explains what engineers must verify before deployment. "
            "The third segment closes with a practical measurement question."
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "NARRATION_SEGMENTS": "3",
                "VOICE_REVIEW_MAX_ATTEMPTS": "3",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "45",
                "AUDIO_SEGMENT_PAUSE_MS": "20",
            },
            clear=True,
        ):
            settings = Settings.from_env()
            tts = ToneTTS(settings.voice_contract.target_wpm)
            reviewer = RejectThenApproveReviewer()
            result = build_reviewed_narration(
                settings,
                narration,
                Path(temporary),
                tts=tts,
                reviewer=reviewer,
            )

        self.assertEqual(result.attempts, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(len(tts.calls), 4)
        attempts = {segment.segment_id: segment.attempt for segment in result.segments}
        self.assertEqual(attempts, {0: 1, 1: 2, 2: 1})
        self.assertIn("exact script fidelity", tts.calls[-1][1])


if __name__ == "__main__":
    unittest.main()
