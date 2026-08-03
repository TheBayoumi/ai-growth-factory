from __future__ import annotations

import unittest

from factory.production_source_deduplication import deduplicate_package_sources


class ProductionSourceDeduplicationTests(unittest.TestCase):
    def test_preserves_first_occurrence_and_invalidates_scene_positions(self) -> None:
        raw = {
            "source_urls": [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/a",
            ],
            "source_publishers": ["Publisher A", "Publisher B", "Publisher A"],
            "scenes": [
                {"heading": f"Scene {index}", "source_index": index % 3}
                for index in range(6)
            ],
        }

        repaired = deduplicate_package_sources(raw)

        self.assertEqual(
            repaired["source_urls"],
            ["https://example.com/a", "https://example.com/b"],
        )
        self.assertEqual(
            repaired["source_publishers"],
            ["Publisher A", "Publisher B"],
        )
        self.assertTrue(
            all(scene["source_index"] == -1 for scene in repaired["scenes"])
        )
        self.assertEqual(raw["source_urls"][2], "https://example.com/a")
        self.assertEqual(raw["scenes"][0]["source_index"], 0)

    def test_unique_source_list_is_returned_unchanged(self) -> None:
        raw = {
            "source_urls": ["https://example.com/a", "https://example.com/b"],
            "source_publishers": ["Publisher A", "Publisher B"],
            "scenes": [{"source_index": 0}] * 6,
        }
        self.assertIs(deduplicate_package_sources(raw), raw)

    def test_mismatched_parallel_arrays_remain_fail_closed(self) -> None:
        raw = {
            "source_urls": ["https://example.com/a", "https://example.com/a"],
            "source_publishers": ["Publisher A"],
            "scenes": [{"source_index": 0}] * 6,
        }
        self.assertIs(deduplicate_package_sources(raw), raw)


if __name__ == "__main__":
    unittest.main()
