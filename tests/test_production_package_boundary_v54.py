from __future__ import annotations

import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.production_package_boundary_v54 import normalize_raw_package_boundary_v54


class ProductionPackageBoundaryV54Tests(unittest.TestCase):
    def test_trivial_scene_overshoot_is_trimmed_without_metadata_changes(self) -> None:
        source = SourceItem(
            "Example",
            "Current AI release",
            "https://example.com/release",
            "A source-backed current release with concrete implementation details.",
            datetime.now(timezone.utc),
            author="Example",
        )
        body = (
            "Engineers connect the inference service to an existing application while measuring "
            "latency and checking deployment behavior before broader production rollout today."
        )
        self.assertGreater(len(body.split()), 18)
        raw = {
            "narration": "word " * 132,
            "source_urls": [source.url],
            "scenes": [
                {
                    "heading": "A heading with too many words here",
                    "body": body,
                    "visual": "An engineer connects an application to an inference service.",
                    "source_index": 0,
                }
            ],
        }

        corrected = normalize_raw_package_boundary_v54(raw, [source])

        scene = corrected["scenes"][0]
        self.assertLessEqual(len(scene["heading"].split()), 5)
        self.assertLessEqual(len(scene["body"].split()), 18)
        self.assertEqual(0, scene["source_index"])
        self.assertEqual(raw["scenes"][0]["visual"], scene["visual"])
        self.assertEqual(raw["source_urls"], corrected["source_urls"])
        self.assertEqual(body, raw["scenes"][0]["body"])

    def test_large_scene_body_is_left_for_strict_validator(self) -> None:
        source = SourceItem(
            "Example",
            "Current AI release",
            "https://example.com/release",
            "Evidence.",
            datetime.now(timezone.utc),
            author="Example",
        )
        body = " ".join(f"word{i}" for i in range(50))
        raw = {"scenes": [{"heading": "Scene", "body": body, "visual": "x", "source_index": 0}]}
        corrected = normalize_raw_package_boundary_v54(raw, [source])
        self.assertEqual(body, corrected["scenes"][0]["body"])


if __name__ == "__main__":
    unittest.main()
