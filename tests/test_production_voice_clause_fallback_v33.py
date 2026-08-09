from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

from factory import production_voice_calibration_v28 as calibration
from factory.production_voice_clause_fallback_v33 import (
    build_clause_fallback_tts_class_v33,
    join_pcm_wavs_v33,
    split_sentence_for_clause_fallback_v33,
)
from factory.qwen_tts import QwenTTSError
from factory.video_profile import VideoProfile


_UNREACHABLE = "Qwen TTS produced no candidate reachable within the v28 1.15x tempo ceiling"


def _write_tone(path: Path, *, duration_seconds: float, sample_rate: int = 8000) -> None:
    frame_count = max(1, round(duration_seconds * sample_rate))
    samples = array(
        "h",
        (
            round(1800 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(frame_count)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


class _FullSentenceFailsBase:
    def __init__(self, settings: object) -> None:
        self.settings = settings

    def generate(
        self,
        *,
        text: str,
        instruction: str,
        output_path: Path,
        seed: int,
    ) -> Path:
        del instruction, seed
        words = len(text.split())
        if words > 10:
            raise QwenTTSError(_UNREACHABLE)
        _write_tone(output_path, duration_seconds=words / 130.0 * 60.0)
        return output_path


class _AlwaysUnreachableBase:
    def __init__(self, settings: object) -> None:
        self.settings = settings

    def generate(self, **kwargs: object) -> Path:
        del kwargs
        raise QwenTTSError(_UNREACHABLE)


class ProductionVoiceClauseFallbackV33Tests(unittest.TestCase):
    def setUp(self) -> None:
        calibration._CALIBRATION_EVENTS.clear()

    def tearDown(self) -> None:
        calibration._CALIBRATION_EVENTS.clear()

    def test_split_preserves_exact_normalized_transcript(self) -> None:
        text = (
            "This framework allows researchers to reuse infrastructure, reducing complexity "
            "while maintaining strong performance across various tasks."
        )
        clauses = split_sentence_for_clause_fallback_v33(text)
        self.assertEqual(len(clauses), 2)
        self.assertEqual(" ".join(clauses), text)
        self.assertTrue(clauses[0].endswith(","))
        self.assertGreaterEqual(min(len(item.split()) for item in clauses), 4)

    def test_pcm_join_adds_only_the_configured_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            second = root / "second.wav"
            output = root / "joined.wav"
            _write_tone(first, duration_seconds=1.0)
            _write_tone(second, duration_seconds=1.0)
            join_pcm_wavs_v33((first, second), output, pause_ms=120)
            with wave.open(str(output), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            self.assertAlmostEqual(duration, 2.12, places=3)

    def test_unreachable_sentence_recovers_from_reachable_clauses(self) -> None:
        fallback_type = build_clause_fallback_tts_class_v33(
            _FullSentenceFailsBase,
            profile=VideoProfile(),
        )
        text = (
            "This framework allows researchers to reuse infrastructure, reducing complexity "
            "while maintaining strong performance across various tasks."
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "work" / "segments" / "segment.wav"
            generated = fallback_type(SimpleNamespace()).generate(
                text=text,
                instruction="Natural technical narration.",
                output_path=output,
                seed=17,
            )
            self.assertEqual(generated, output)
            self.assertTrue(output.is_file())
            events = [
                event
                for event in calibration._CALIBRATION_EVENTS
                if event.get("type") == "bounded_clause_fallback_v33"
            ]
            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]["reachable"])
            self.assertEqual(events[0]["clause_count"], 2)
            self.assertGreater(float(events[0]["combined_observed_wpm"]), 120.0)

    def test_failed_fallback_persists_candidate_diagnostics(self) -> None:
        fallback_type = build_clause_fallback_tts_class_v33(
            _AlwaysUnreachableBase,
            profile=VideoProfile(),
        )
        text = (
            "This framework allows researchers to reuse infrastructure, reducing complexity "
            "while maintaining strong performance across various tasks."
        )
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary) / "work"
            output = workdir / "segments" / "segment.wav"
            with self.assertRaises(QwenTTSError):
                fallback_type(SimpleNamespace()).generate(
                    text=text,
                    instruction="Natural technical narration.",
                    output_path=output,
                    seed=23,
                )
            diagnostic = workdir / "voice-calibration-failure.json"
            self.assertTrue(diagnostic.is_file())
            payload = json.loads(diagnostic.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "voice_clause_fallback_failed_closed")
            self.assertEqual(payload["text"], text)
            self.assertEqual(len(payload["clauses"]), 2)
            self.assertIn("no candidate reachable", payload["fallback_error"].casefold())


if __name__ == "__main__":
    unittest.main()
