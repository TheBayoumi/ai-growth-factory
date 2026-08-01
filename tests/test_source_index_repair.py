import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.source_index_repair import _production_normalizer


class SourceIndexRepairTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.sources = [
            SourceItem(
                "OpenAI",
                "Realtime model release",
                "https://example.com/realtime",
                "The release improves low latency voice interaction.",
                now,
            ),
            SourceItem(
                "NVIDIA",
                "New inference hardware",
                "https://example.com/hardware",
                "Hardware improves inference throughput and efficiency.",
                now,
            ),
        ]
        self.urls = [source.url for source in self.sources]

    @staticmethod
    def failing_original(*_args):
        raise RuntimeError("Scene source_index out of range: 5")

    def test_scene_ordinal_pattern_is_repaired_to_selected_sources(self):
        scenes = [
            {
                "heading": "Realtime voice",
                "body": "Lower latency improves voice interaction.",
                "visual": "Audio latency diagram.",
                "source_index": index,
            }
            for index in range(6)
        ]
        repaired = _production_normalizer(
            self.failing_original,
            scenes,
            self.urls,
            self.sources,
        )
        self.assertEqual(repaired, [0, 0, 0, 0, 0, 0])

    def test_valid_indices_remain_unchanged_when_one_invalid_scene_is_repaired(self):
        scenes = [
            {
                "heading": "Inference hardware" if index == 5 else "Realtime voice",
                "body": "Hardware throughput." if index == 5 else "Voice latency.",
                "visual": "Diagram.",
                "source_index": 5 if index == 5 else 0,
            }
            for index in range(6)
        ]
        repaired = _production_normalizer(
            self.failing_original,
            scenes,
            self.urls,
            self.sources,
        )
        self.assertEqual(repaired[:5], [0, 0, 0, 0, 0])
        self.assertEqual(repaired[5], 1)

    def test_non_source_validation_failure_is_not_suppressed(self):
        def original(*_args):
            raise RuntimeError("Narration word count outside quality gate")

        with self.assertRaisesRegex(RuntimeError, "Narration word count"):
            _production_normalizer(original, [], self.urls, self.sources)


if __name__ == "__main__":
    unittest.main()
