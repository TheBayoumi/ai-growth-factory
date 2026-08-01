import json
import math
import tempfile
import unittest
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.models import AudioMetrics, AudioReview
from factory.voice_pipeline import _global_repair, build_reviewed_narration


class FakeTTS:
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


class FakeReviewer:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _payload(decision: str, failed: list[dict[str, object]]) -> dict[str, object]:
        passing = decision == "approve"
        return {
            "decision": decision,
            "overall_score": 0.92 if passing else 0.82,
            "scores": {
                "script_fidelity": 0.99,
                "naturalness": 0.91 if passing else 0.80,
                "authority": 0.91,
                "engagement": 0.90,
                "pronunciation": 0.96,
                "pace": 0.90,
                "pause_quality": 0.90,
                "emotional_match": 0.90,
                "audio_artifacts": 0.96,
            },
            "failed_segments": failed,
            "summary": "Approved." if passing else "Segment one needs more energy.",
        }

    def review(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls == 1:
            payload = self._payload(
                "retry_segments",
                [
                    {
                        "segment_id": 1,
                        "reason": "Flat delivery.",
                        "tts_instruction": "Increase energy and shorten the opening pause.",
                    }
                ],
            )
        else:
            payload = self._payload("approve", [])
        return AudioReview.from_dict(payload, model="gpt-realtime-2.1")


class VoicePipelineTests(unittest.TestCase):
    def test_only_rejected_segment_is_regenerated(self):
        narration = (
            "Artificial intelligence changed this week with a practical new capability. "
            "The evidence matters because benchmark headlines can hide real limitations. "
            "Here is what engineers should test before trusting the new workflow."
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "NARRATION_SEGMENTS": "3",
                "VOICE_REVIEW_MAX_ATTEMPTS": "3",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "40",
                "AUDIO_SEGMENT_PAUSE_MS": "50",
            },
            clear=True,
        ):
            settings = Settings.from_env()
            tts = FakeTTS(settings.voice_contract.target_wpm)
            reviewer = FakeReviewer()
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
            self.assertTrue(result.audio_path.exists())
            self.assertTrue(result.manifest_path.exists())
            attempts = {segment.segment_id: segment.attempt for segment in result.segments}
            self.assertEqual(attempts, {0: 1, 1: 2, 2: 1})

    def test_locally_weak_approve_result_retries_all_segments(self):
        class WeakThenStrongReviewer:
            def __init__(self):
                self.calls = 0

            def review(self, **kwargs):
                del kwargs
                self.calls += 1
                payload = FakeReviewer._payload("approve", [])
                if self.calls == 1:
                    payload["overall_score"] = 0.82
                    payload["scores"]["naturalness"] = 0.70
                return AudioReview.from_dict(payload, model="qwen-omni-test")

        narration = (
            "The first segment introduces a useful artificial intelligence update. "
            "The second segment explains what engineers should verify before deployment."
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "NARRATION_SEGMENTS": "2",
                "VOICE_REVIEW_MAX_ATTEMPTS": "3",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "45",
                "AUDIO_SEGMENT_PAUSE_MS": "20",
            },
            clear=True,
        ):
            settings = Settings.from_env()
            tts = FakeTTS(settings.voice_contract.target_wpm)
            reviewer = WeakThenStrongReviewer()
            result = build_reviewed_narration(
                settings, narration, Path(temporary), tts=tts, reviewer=reviewer
            )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(len(tts.calls), 4)
        self.assertTrue(all("human phrasing" in call[1] for call in tts.calls[2:]))

    def test_review_can_be_disabled_for_local_smoke_test(self):
        narration = "First sentence has enough words for testing. Second sentence also has enough words for testing."
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "NARRATION_SEGMENTS": "2",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "45",
                "AUDIO_SEGMENT_PAUSE_MS": "20",
            },
            clear=True,
        ):
            settings = replace(Settings.from_env(), reviewer_required=False)
            tts = FakeTTS(settings.voice_contract.target_wpm)
            result = build_reviewed_narration(settings, narration, Path(temporary), tts=tts)
            self.assertIsNone(result.review)
            self.assertEqual(result.attempts, 1)

    def test_bounded_tempo_correction_accepts_the_117_wpm_canary_case(self):
        narration = (
            "This practical update explains the new capability and why its evidence matters for engineers. "
            "Before deployment, compare the primary sources, test a controlled task, and measure the result."
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "NARRATION_SEGMENTS": "2",
                "VOICE_REVIEW_MAX_ATTEMPTS": "3",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "32",
                "AUDIO_SEGMENT_PAUSE_MS": "0",
            },
            clear=True,
        ):
            settings = replace(Settings.from_env(), reviewer_required=False)
            tts = FakeTTS(117)
            result = build_reviewed_narration(settings, narration, Path(temporary), tts=tts)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(tts.calls), 2)
        self.assertTrue(result.metrics.passed, result.metrics.failures)
        self.assertGreaterEqual(result.metrics.estimated_wpm, 123.0)
        self.assertLessEqual(result.metrics.estimated_wpm, 187.0)
        corrections = [
            review for review in manifest["reviews"]
            if review["type"] == "deterministic_tempo_correction"
        ]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["factor"], 1.15)
        self.assertEqual(corrections[0]["decision"], "accept")
        self.assertAlmostEqual(
            result.segments[-1].end_seconds,
            result.metrics.duration_seconds,
            delta=0.15,
        )

    def test_global_pace_repair_is_quantified(self):
        metrics = AudioMetrics(
            duration_seconds=60.0,
            sample_rate=24000,
            channels=1,
            peak_dbfs=-3.0,
            rms_dbfs=-20.0,
            clipping_ratio=0.0,
            silence_ratio=0.0,
            max_silence_seconds=0.0,
            estimated_wpm=117.0,
            dc_offset=0.0,
            passed=False,
            failures=("estimated pace 117.0 WPM is outside target 155±32",),
        )

        repair = _global_repair(metrics, 155)

        self.assertIn("about 32%", repair)
        self.assertIn("155 words per minute", repair)


if __name__ == "__main__":
    unittest.main()
