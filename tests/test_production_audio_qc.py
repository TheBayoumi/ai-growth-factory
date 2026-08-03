from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from factory.audio_qc import normalize_audio
from factory.models import FailedSegment
from factory.production_audio_qc import (
    deterministic_qc_failure,
    normalize_audio_with_headroom,
    production_peak_target,
)
from factory.voice_pipeline import VoiceGenerationError


def _write_hot_sine(path: Path, *, seconds: float = 3.0) -> None:
    sample_rate = 24000
    samples = array("h")
    for index in range(round(sample_rate * seconds)):
        value = round(0.95 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
        samples.append(value)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def _peak_dbfs(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    peak = max(abs(sample) for sample in samples) / 32768.0
    return 20 * math.log10(max(peak, 1e-12))


class ProductionAudioQCTests(unittest.TestCase):
    def test_production_peak_target_adds_half_db_headroom(self) -> None:
        self.assertAlmostEqual(production_peak_target(-1.0), -1.5)

    def test_real_ffmpeg_normalization_lands_inside_unchanged_peak_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "hot.wav"
            output = root / "normalized.wav"
            _write_hot_sine(source)

            normalize_audio_with_headroom(
                source,
                output,
                target_lufs=-16.0,
                peak_dbfs=-1.0,
                normalizer=normalize_audio,
            )

            measured = _peak_dbfs(output)
            self.assertLessEqual(measured, -1.0, measured)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 44)

    def test_final_deterministic_qc_replaces_stale_reviewer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "voice-review-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "attempts": 3,
                        "metrics": {
                            "passed": False,
                            "failures": [
                                "peak -0.86 dBFS exceeds limit -1.00 dBFS"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            stale = VoiceGenerationError(
                "segment 4 needs clearer articulation",
                manifest_path=manifest,
                failed_segments=(
                    FailedSegment(
                        4,
                        "inconsistent pace and unclear articulation",
                        "Please try again with clearer articulation.",
                    ),
                ),
            )

            rewritten = deterministic_qc_failure(stale)

            self.assertIsNot(rewritten, stale)
            self.assertIsInstance(rewritten, VoiceGenerationError)
            self.assertIn("failed deterministic QC after 3 attempts", str(rewritten))
            self.assertIn("peak -0.86 dBFS exceeds limit -1.00 dBFS", str(rewritten))
            self.assertNotIn("segment 4", str(rewritten))
            self.assertEqual(rewritten.failed_segments, ())
            self.assertEqual(rewritten.manifest_path, manifest)

    def test_reviewer_failure_is_preserved_when_final_qc_passed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "voice-review-manifest.json"
            manifest.write_text(
                json.dumps({"attempts": 3, "metrics": {"passed": True, "failures": []}}),
                encoding="utf-8",
            )
            original = VoiceGenerationError(
                "segment 5 remains not fluent",
                manifest_path=manifest,
            )

            self.assertIs(deterministic_qc_failure(original), original)


if __name__ == "__main__":
    unittest.main()
