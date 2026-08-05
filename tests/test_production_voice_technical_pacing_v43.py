from __future__ import annotations

import unittest
from pathlib import Path

from factory.production_voice_technical_pacing_v43 import technical_editorial_factor_v43
from factory.production_voice_technical_identifier_v42 import (
    speech_equivalent_word_count_v42,
    speech_equivalent_wpm_v42,
)
from factory.video_profile import VideoProfile


_FAILED_SEGMENT = (
    "The model, LFM2.5-2.6B, allows developers to run AI tasks without relying on cloud infrastructure."
)


class ProductionVoiceTechnicalPacingV43Tests(unittest.TestCase):
    def test_exact_failed_segment_enters_editorial_range_at_existing_ceiling(self) -> None:
        profile = VideoProfile()
        written_wpm = 103.19
        duration = len(_FAILED_SEGMENT.split()) / written_wpm * 60.0
        effective_wpm = speech_equivalent_wpm_v42(_FAILED_SEGMENT, duration)
        result = technical_editorial_factor_v43(
            _FAILED_SEGMENT,
            duration,
            profile=profile,
        )

        self.assertEqual(len(_FAILED_SEGMENT.split()), 14)
        self.assertEqual(speech_equivalent_word_count_v42(_FAILED_SEGMENT), 16.0)
        self.assertAlmostEqual(effective_wpm, 117.931, places=3)
        self.assertIsNotNone(result)
        assert result is not None
        factor, projected = result
        self.assertEqual(factor, profile.maximum_tempo_factor)
        self.assertGreaterEqual(projected, 132.0)
        self.assertLessEqual(projected, 152.0)

    def test_plain_sentence_keeps_written_word_pacing(self) -> None:
        text = "An ordinary sentence uses the standard editorial timing calculation."
        duration = len(text.split()) / 140.0 * 60.0
        result = technical_editorial_factor_v43(
            text,
            duration,
            profile=VideoProfile(),
        )
        self.assertIsNotNone(result)
        assert result is not None
        _factor, projected = result
        self.assertAlmostEqual(projected, 144.0, places=3)

    def test_runtime_installs_v43_after_v42(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        identifier = source.index("install_production_voice_technical_identifier_v42()")
        pacing = source.index("install_production_voice_technical_pacing_v43()")
        self.assertGreater(pacing, identifier)


if __name__ == "__main__":
    unittest.main()
