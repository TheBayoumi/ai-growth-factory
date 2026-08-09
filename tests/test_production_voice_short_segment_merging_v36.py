from __future__ import annotations

import unittest

from factory.production_voice_capacity_v29 import split_narration_for_voice_v29


class ProductionVoiceShortSegmentMergingV36Tests(unittest.TestCase):
    def test_failed_ten_word_technical_sentence_is_not_synthesized_alone(self) -> None:
        narration = (
            "The model, LFM2.5-2.6B, is designed for efficient and secure deployment. "
            "It runs locally on compact hardware while preserving private data."
        )

        segments = split_narration_for_voice_v29(narration, target_segments=1)

        self.assertEqual(segments, [narration])
        self.assertGreaterEqual(len(segments[0].split()), 12)
        self.assertLessEqual(len(segments[0].split()), 24)

    def test_transcript_is_preserved_exactly_when_short_sentences_merge(self) -> None:
        narration = (
            "A compact model runs locally. "
            "The system keeps private data on device. "
            "Engineers measure latency and repeatability before deployment."
        )

        segments = split_narration_for_voice_v29(narration, target_segments=2)

        self.assertEqual(" ".join(segments), narration)
        self.assertTrue(all(12 <= len(segment.split()) <= 24 for segment in segments))


if __name__ == "__main__":
    unittest.main()
