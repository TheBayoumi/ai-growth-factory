import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.models import NarrationSegment, VoiceContract
from factory.voice_pipeline import _pace_correct_segment_assets


class VoiceSegmentPacingTests(unittest.TestCase):
    def test_paced_asset_replaces_raw_segment_before_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "segments" / "segment-00-attempt-1.wav"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"raw")
            segment = NarrationSegment(
                segment_id=0,
                text=" ".join(f"word{index}" for index in range(100)),
                instruction="Narrate clearly.",
                audio_path=raw,
                attempt=1,
            )
            settings = SimpleNamespace(
                audio_wpm_tolerance=15,
                audio_target_lufs=-16.0,
                audio_peak_limit_dbfs=-1.0,
            )

            def fake_correct(source, destination, *, factor):
                self.assertEqual(source, raw)
                self.assertEqual(factor, 1.4)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"paced")
                return destination

            def fake_normalize(source, destination, *, target_lufs, peak_dbfs):
                self.assertTrue(source.name.endswith("paced-1.wav"))
                self.assertEqual(target_lufs, -16.0)
                self.assertEqual(peak_dbfs, -1.0)
                destination.write_bytes(b"normalized")
                return destination

            with (
                patch("factory.voice_pipeline.wav_duration", side_effect=[60.0, 42.857]),
                patch("factory.voice_pipeline.tempo_correction_factor", return_value=1.4),
                patch("factory.voice_pipeline.correct_audio_tempo", side_effect=fake_correct),
                patch("factory.voice_pipeline.normalize_audio", side_effect=fake_normalize),
            ):
                corrected, events = _pace_correct_segment_assets(
                    [segment],
                    workdir=root,
                    pipeline_attempt=1,
                    contract=VoiceContract(target_wpm=155),
                    settings=settings,
                )

            self.assertNotEqual(corrected[0].audio_path, raw)
            self.assertTrue(corrected[0].audio_path.name.endswith("paced-normalized-1.wav"))
            self.assertEqual(events[0]["type"], "deterministic_segment_tempo_correction")
            self.assertEqual(events[0]["segment_id"], 0)
            self.assertAlmostEqual(events[0]["before_wpm"], 100.0)
            self.assertAlmostEqual(events[0]["after_wpm"], 140.001, places=2)

    def test_already_compliant_asset_is_preserved_without_reprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paced = root / "segment-00-paced.wav"
            paced.write_bytes(b"paced")
            segment = NarrationSegment(0, "one two three", "instruction", paced)
            settings = SimpleNamespace(
                audio_wpm_tolerance=15,
                audio_target_lufs=-16.0,
                audio_peak_limit_dbfs=-1.0,
            )
            with (
                patch("factory.voice_pipeline.wav_duration", return_value=1.2),
                patch("factory.voice_pipeline.tempo_correction_factor", return_value=None),
                patch("factory.voice_pipeline.correct_audio_tempo") as correct,
                patch("factory.voice_pipeline.normalize_audio") as normalize,
            ):
                corrected, events = _pace_correct_segment_assets(
                    [segment],
                    workdir=root,
                    pipeline_attempt=2,
                    contract=VoiceContract(target_wpm=155),
                    settings=settings,
                )
            self.assertEqual(corrected, [segment])
            self.assertEqual(events, [])
            correct.assert_not_called()
            normalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
