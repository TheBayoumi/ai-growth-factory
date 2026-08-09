from __future__ import annotations

import unittest
from pathlib import Path

from factory.production_voice_technical_identifier_v42 import (
    pace_multiplier_v42,
    speech_equivalent_word_count_v42,
    speech_equivalent_wpm_v42,
    technical_token_weight_v42,
)
from factory.video_profile import VideoProfile


_FAILED_SENTENCE = (
    "Hugging Face has launched LFM2.5-2.6B, an AI model designed for efficient local deployment."
)
_NUMERIC_FAILED_SENTENCE = (
    "The system supports 81 organizational groups and has driven 98.6% higher productivity."
)


class ProductionVoiceTechnicalIdentifierV42Tests(unittest.TestCase):
    def test_failed_identifier_receives_bounded_spoken_weight(self) -> None:
        self.assertEqual(len(_FAILED_SENTENCE.split()), 13)
        self.assertEqual(technical_token_weight_v42("LFM2.5-2.6B"), 3.0)
        self.assertEqual(speech_equivalent_word_count_v42(_FAILED_SENTENCE), 15.0)
        self.assertAlmostEqual(pace_multiplier_v42(_FAILED_SENTENCE), 15 / 13, places=6)

    def test_exact_failed_take_is_reachable_without_relaxing_tempo_ceiling(self) -> None:
        profile = VideoProfile()
        effective_observed = 112.392 * pace_multiplier_v42(_FAILED_SENTENCE)
        required_factor = profile.target_wpm / effective_observed

        self.assertGreaterEqual(
            effective_observed * profile.maximum_tempo_factor,
            profile.minimum_wpm,
        )
        self.assertLessEqual(required_factor, profile.maximum_tempo_factor)
        self.assertAlmostEqual(required_factor, 1.095, places=3)

    def test_numeric_tokens_count_the_words_qwen_actually_speaks(self) -> None:
        self.assertEqual(technical_token_weight_v42("81"), 2.0)
        self.assertEqual(technical_token_weight_v42("98.6%"), 5.0)
        self.assertEqual(technical_token_weight_v42("84%"), 3.0)
        self.assertEqual(technical_token_weight_v42("500,000+"), 4.0)
        self.assertEqual(technical_token_weight_v42("$20"), 2.0)

    def test_v50_failed_numeric_take_is_reachable_with_correct_denominator(self) -> None:
        profile = VideoProfile()
        written_wpm_from_live_failure = 93.86883380474977
        self.assertEqual(len(_NUMERIC_FAILED_SENTENCE.split()), 12)
        self.assertEqual(
            speech_equivalent_word_count_v42(_NUMERIC_FAILED_SENTENCE),
            17.0,
        )
        multiplier = pace_multiplier_v42(_NUMERIC_FAILED_SENTENCE)
        effective_observed = written_wpm_from_live_failure * multiplier
        required_factor = profile.target_wpm / effective_observed

        self.assertAlmostEqual(multiplier, 17 / 12, places=6)
        self.assertAlmostEqual(effective_observed, 132.98084789006217, places=6)
        self.assertGreaterEqual(
            effective_observed * profile.maximum_tempo_factor,
            profile.minimum_wpm,
        )
        self.assertLessEqual(required_factor, profile.maximum_tempo_factor)
        self.assertAlmostEqual(required_factor, 1.067823, places=6)

        duration_seconds = 12 / written_wpm_from_live_failure * 60.0
        self.assertAlmostEqual(
            speech_equivalent_wpm_v42(_NUMERIC_FAILED_SENTENCE, duration_seconds),
            effective_observed,
            places=6,
        )

    def test_plain_language_word_count_is_unchanged(self) -> None:
        text = "This ordinary sentence contains no mixed technical identifiers."
        self.assertEqual(speech_equivalent_word_count_v42(text), len(text.split()))
        self.assertEqual(pace_multiplier_v42(text), 1.0)

    def test_plain_acronym_does_not_inflate_publication_pace(self) -> None:
        self.assertEqual(technical_token_weight_v42("AI"), 1.0)
        self.assertEqual(technical_token_weight_v42("LLM"), 1.0)

    def test_runtime_installs_v42_after_clause_fallback(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        clause = source.index("install_production_voice_clause_fallback_v33()")
        technical = source.index("install_production_voice_technical_identifier_v42()")
        self.assertGreater(technical, clause)


if __name__ == "__main__":
    unittest.main()
