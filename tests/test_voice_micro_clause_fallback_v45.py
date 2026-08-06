from __future__ import annotations

import unittest

from factory.production_voice_micro_clause_fallback_v45 import (
    choose_piece_to_split_v45,
    combined_micro_clause_wpm_v45,
    split_micro_clause_v45,
)


class VoiceMicroClauseFallbackV45Tests(unittest.TestCase):
    def test_split_is_balanced_and_transcript_exact(self) -> None:
        text = "Alpha beta gamma delta and epsilon zeta eta theta"
        pieces = split_micro_clause_v45(text)
        self.assertEqual(2, len(pieces))
        self.assertGreaterEqual(len(pieces[0].split()), 4)
        self.assertGreaterEqual(len(pieces[1].split()), 4)
        self.assertEqual(text, " ".join(pieces))

    def test_short_phrase_is_not_split(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta"
        self.assertEqual((text,), split_micro_clause_v45(text))

    def test_slowest_safely_splittable_piece_is_selected(self) -> None:
        pieces = (
            "One two three four five six seven eight",
            "Nine ten eleven twelve thirteen fourteen fifteen sixteen",
        )
        self.assertEqual(
            1,
            choose_piece_to_split_v45(pieces, (110.0, 90.0), maximum_pieces=4),
        )
        self.assertIsNone(
            choose_piece_to_split_v45(
                ("one two three four",) * 4,
                (90.0,) * 4,
                maximum_pieces=4,
            )
        )

    def test_combined_pace_uses_spoken_equivalent_identifier_weight(self) -> None:
        text = "Model LFM2.5-2.6B improves local planning"
        written_wpm = len(text.split()) / 3.0 * 60.0
        spoken_equivalent_wpm = combined_micro_clause_wpm_v45(text, 3.0)
        self.assertGreater(spoken_equivalent_wpm, written_wpm)

    def test_runtime_installer_wraps_current_voice_class(self) -> None:
        from factory import canary, production_voice_micro_clause_fallback_v45 as module
        from factory import voice_pipeline

        original_tts = voice_pipeline.Qwen3TTS
        original_copy = canary._copy_voice_diagnostics
        original_installed = module._INSTALLED
        try:
            module._INSTALLED = False
            module.install_production_voice_micro_clause_fallback_v45()
            self.assertIsNot(voice_pipeline.Qwen3TTS, original_tts)
            self.assertTrue(issubclass(voice_pipeline.Qwen3TTS, original_tts))
            self.assertIsNot(canary._copy_voice_diagnostics, original_copy)
        finally:
            voice_pipeline.Qwen3TTS = original_tts
            canary._copy_voice_diagnostics = original_copy
            module._INSTALLED = original_installed


if __name__ == "__main__":
    unittest.main()
