from __future__ import annotations

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
from factory.models import (
    AudioMetrics,
    AudioReview,
    FailedSegment,
    ReviewScores,
)
from factory.voice_pipeline import _global_repair, build_reviewed_narration


class FakeTTS:
    def __init__(self, words_per_minute: int = 155) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.words_per_minute = words_per_minute

    def generate(self, *, text: str, instruction: str, output_path: Path, seed: int) -> Path:
        self.calls.append((text, instruction, seed))
        words = max(1, len(text.split()))
        duration = words / self.words_per_minute * 60
        sample_rate = 24000
        frame_count = max(1, int(sample_rate * duration))
        samples = array("h")
        for index in range(frame_count):
            value = int(10000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            samples.append(value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())
        return output_path

    def unload(self) -> None:
        return None


def _scores(value: float = 0.95, *, naturalness: float | None = None) -> ReviewScores:
    return ReviewScores(
        script_fidelity=0.99,
        naturalness=value if naturalness is None else naturalness,
        authority=value,
        engagement=value,
        pronunciation=0.95,
        pace=value,
        pause_quality=value,
        emotional_match=value,
        audio_artifacts=0.95,
    )


class RetryOneReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, **kwargs) -> AudioReview:
        self.calls += 1
        if self.calls == 1:
            return AudioReview(
                decision="retry_segments",
                overall_score=0.82,
                scores=_scores(0.90, naturalness=0.80),
                failed_segments=(
                    FailedSegment(1, "Pacing is flat", "Use more dynamic but natural pacing"),
                ),
                summary="One segment needs repair",
                reviewer_model="fixture",
            )
        return AudioReview(
            decision="approve",
            overall_score=0.95,
            scores=_scores(),
            failed_segments=(),
            summary="Approved",
            reviewer_model="fixture",
        )

    def unload(self) -> None:
        return None


class WeakThenStrongReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, **kwargs) -> AudioReview:
        self.calls += 1
        if self.calls == 1:
            return AudioReview(
                decision="approve",
                overall_score=0.90,
                scores=_scores(0.90, naturalness=0.70),
                failed_segments=(),
                summary="Looks acceptable but is locally weak",
                reviewer_model="fixture",
            )
        return AudioReview(
            decision="approve",
            overall_score=0.97,
            scores=_scores(),
            failed_segments=(),
            summary="Approved",
            reviewer_model="fixture",
        )

    def unload(self) -> None:
        return None


class VoicePipelineTests(unittest.TestCase):
    def test_only_rejected_segment_is_regenerated(self):
        narration = "First sentence has enough words for testing. Second sentence also has enough words for testing."
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
            reviewer = RetryOneReviewer()
            result = build_reviewed_narration(
                settings, narration, Path(temporary), tts=tts, reviewer=reviewer
            )
        self.assertEqual(result.attempts, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(len(tts.calls), 3)
        self.assertNotEqual(tts.calls[0][0], tts.calls[2][0])
        self.assertEqual(tts.calls[1][0], tts.calls[2][0])
        self.assertIn("dynamic but natural pacing", tts.calls[2][1])

    def test_locally_weak_approve_result_retries_all_segments(self):
        narration = "First sentence has enough words for testing. Second sentence also has enough words for testing."
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
        segment_corrections = [
            review
            for review in manifest["reviews"]
            if review["type"] == "deterministic_segment_tempo_correction"
        ]
        self.assertEqual(len(segment_corrections), 2)
        self.assertEqual({item["segment_id"] for item in segment_corrections}, {0, 1})
        self.assertTrue(all(1.0 < item["factor"] <= 1.45 for item in segment_corrections))
        self.assertTrue(all(item["after_wpm"] >= 123.0 for item in segment_corrections))
        track_corrections = [
            review
            for review in manifest["reviews"]
            if review["type"] == "deterministic_track_tempo_correction"
        ]
        self.assertEqual(track_corrections, [])
        self.assertTrue(all("paced-normalized" in segment.audio_path.name for segment in result.segments))
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
