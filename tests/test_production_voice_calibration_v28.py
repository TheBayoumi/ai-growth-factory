from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from factory.production_voice_bounds_v28 import bounded_tempo_factor_v28
from factory.production_voice_calibration_v28 import (
    calibrated_segment_band_v28,
    compact_excess_silence_v28,
    segment_candidate_reachable_v28,
    synthesis_target_for_observation_v28,
)
from factory.video_profile import VideoProfile


class ProductionVoiceCalibrationV28Tests(unittest.TestCase):
    @staticmethod
    def _tone(sample_rate: int, seconds: float, frequency: float = 220.0) -> array:
        count = round(sample_rate * seconds)
        return array(
            "h",
            (
                round(9000 * math.sin(2 * math.pi * frequency * index / sample_rate))
                for index in range(count)
            ),
        )

    @staticmethod
    def _write(path: Path, samples: array, sample_rate: int = 24000) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())

    @staticmethod
    def _duration(path: Path) -> float:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()

    def test_compacts_long_internal_pause_without_touching_speech_duration(self) -> None:
        sample_rate = 24000
        first = self._tone(sample_rate, 0.60)
        pause = array("h", [0]) * round(sample_rate * 0.90)
        second = self._tone(sample_rate, 0.60, 330.0)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            output = Path(temporary) / "compacted.wav"
            self._write(source, array("h", [*first, *pause, *second]), sample_rate)
            event = compact_excess_silence_v28(source, output)

            self.assertAlmostEqual(event["before_seconds"], 2.10, places=2)
            self.assertAlmostEqual(event["after_seconds"], 1.42, delta=0.04)
            self.assertAlmostEqual(event["removed_seconds"], 0.68, delta=0.04)
            self.assertLessEqual(event["longest_silence_after_seconds"], 0.22)
            self.assertGreater(output.stat().st_size, 1000)

    def test_compaction_makes_a_slow_track_reachable_below_115_percent(self) -> None:
        # The failed Modal track measured 138 words over 71.38 seconds (116 WPM).
        # Removing only excessive dead air projects roughly 65.98 seconds, which is
        # 125.5 WPM and therefore reaches 142 WPM with about 1.13x correction.
        compacted_wpm = 138 / 65.98 * 60.0
        factor = bounded_tempo_factor_v28(
            estimated_wpm=compacted_wpm,
            target_wpm=142,
            tolerance=4,
        )
        self.assertIsNotNone(factor)
        assert factor is not None
        self.assertLessEqual(factor, 1.15)
        self.assertGreaterEqual(compacted_wpm * factor, 138.0)
        self.assertLessEqual(compacted_wpm * factor, 146.0)

    def test_short_natural_pause_is_preserved(self) -> None:
        sample_rate = 24000
        first = self._tone(sample_rate, 0.25)
        pause = array("h", [0]) * round(sample_rate * 0.16)
        second = self._tone(sample_rate, 0.25, 330.0)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            output = Path(temporary) / "compacted.wav"
            self._write(source, array("h", [*first, *pause, *second]), sample_rate)
            event = compact_excess_silence_v28(source, output)
            self.assertAlmostEqual(event["removed_seconds"], 0.0, delta=0.025)
            self.assertAlmostEqual(self._duration(output), self._duration(source), delta=0.025)

    def test_segment_band_leaves_headroom_for_inter_segment_pauses(self) -> None:
        self.assertEqual(calibrated_segment_band_v28(VideoProfile()), (140.0, 144.0))

    def test_failed_slow_segments_are_regenerated_instead_of_globally_speeding_track(self) -> None:
        profile = VideoProfile()
        self.assertFalse(segment_candidate_reachable_v28(116.2, profile=profile))
        self.assertFalse(segment_candidate_reachable_v28(105.3, profile=profile))
        self.assertEqual(synthesis_target_for_observation_v28(116.2, profile=profile), 174)
        self.assertEqual(synthesis_target_for_observation_v28(105.3, profile=profile), 185)

    def test_good_and_safely_slowable_segments_are_retained(self) -> None:
        profile = VideoProfile()
        self.assertTrue(segment_candidate_reachable_v28(142.0, profile=profile))
        self.assertTrue(segment_candidate_reachable_v28(160.9, profile=profile))
        self.assertFalse(segment_candidate_reachable_v28(190.0, profile=profile))
        self.assertEqual(synthesis_target_for_observation_v28(190.0, profile=profile), 115)


if __name__ == "__main__":
    unittest.main()
