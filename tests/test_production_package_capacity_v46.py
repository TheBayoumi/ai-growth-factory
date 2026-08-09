from __future__ import annotations

import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_package_capacity_v46 import (
    _augment_repair_prompt,
    stabilize_package_capacity,
    stabilize_raw_package_capacity,
    stabilize_raw_scene_capacity,
)


class ProductionPackageCapacityV46Tests(unittest.TestCase):
    @staticmethod
    def _source() -> SourceItem:
        return SourceItem(
            "OpenAI",
            "GPT-5.6 improves coding reliability",
            "https://example.com/gpt-5-6",
            (
                "OpenAI reports a 42 percent reduction in failed tool calls and a "
                "128000-token context window for controlled enterprise evaluations. "
            )
            * 3,
            datetime.now(timezone.utc),
            author="OpenAI",
        )

    @staticmethod
    def _package(narration: str) -> VideoPackage:
        return VideoPackage(
            topic="GPT-5.6 coding reliability",
            narration=narration,
            title="OpenAI details GPT-5.6 coding reliability improvements",
            description="Source-backed evaluation.",
            tags=["OpenAI", "GPT-5.6", "coding", "reliability", "AI", "tools", "tests", "evidence"],
            thumbnail_text="GPT-5.6 TESTED",
            top_comment="What would you verify first?",
            scenes=[
                Scene(
                    heading=f"Scene {index}",
                    body=f"OpenAI reports measured coding reliability evidence for controlled workflow test {index}.",
                    visual="A controlled software test bench.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/gpt-5-6"],
            source_publishers=["OpenAI"],
        )

    @staticmethod
    def _oversized_narration() -> tuple[str, str, str]:
        opening = (
            "OpenAI says GPT-5.6 reduced failed tool calls by 42 percent "
            "during controlled coding evaluations."
        )
        measurement = (
            "The reported 128000-token context window lets engineers retain "
            "more repository evidence during long tasks."
        )
        middle = [
            "Engineering teams compare repeated workflow results before changing production systems today."
            for _ in range(16)
        ]
        closing = (
            "Before adopting it, verify the linked evidence and compare failures "
            "against your current workflow."
        )
        narration = " ".join([opening, measurement, *middle, closing])
        return narration, opening, closing

    def test_scene_body_compaction_preserves_metadata(self) -> None:
        raw = {
            "scenes": [
                {
                    "heading": "Measured result",
                    "body": (
                        "OpenAI reports lower tool failure rates in controlled coding workflows, "
                        "while engineers verify the result against repeated production tasks today."
                    ),
                    "visual": "An engineer checks a test log.",
                    "source_index": 2,
                }
            ]
        }

        corrected = stabilize_raw_scene_capacity(raw)

        self.assertLessEqual(len(corrected["scenes"][0]["body"].split()), 18)
        self.assertEqual(corrected["scenes"][0]["source_index"], 2)
        self.assertTrue(corrected["scenes"][0]["body"].endswith("."))
        self.assertEqual(len(raw["scenes"][0]["body"].split()), 20)

    def test_raw_oversized_narration_converges_before_package_validation(self) -> None:
        narration, opening, closing = self._oversized_narration()
        self.assertGreater(len(narration.split()), 190)
        raw = {
            "narration": narration,
            "source_urls": ["https://example.com/gpt-5-6"],
            "scenes": [
                {
                    "heading": "Measured result",
                    "body": (
                        "OpenAI reports lower tool failure rates in controlled coding workflows, "
                        "while engineers verify the result against repeated production tasks today."
                    ),
                    "visual": "An engineer checks a test log.",
                    "source_index": 0,
                }
            ],
        }

        corrected = stabilize_raw_package_capacity(raw, [self._source()])

        count = len(corrected["narration"].split())
        self.assertGreaterEqual(count, 130)
        self.assertLessEqual(count, 140)
        self.assertTrue(corrected["narration"].startswith(opening))
        self.assertTrue(corrected["narration"].endswith(closing))
        self.assertIn("42 percent", corrected["narration"])
        self.assertIn("128000-token", corrected["narration"])
        self.assertLessEqual(len(corrected["scenes"][0]["body"].split()), 18)

    def test_oversized_narration_keeps_hook_measurements_and_close(self) -> None:
        narration, opening, closing = self._oversized_narration()
        self.assertGreater(len(narration.split()), 140)

        corrected = stabilize_package_capacity(
            self._package(narration),
            [self._source()],
        )

        count = len(corrected.narration.split())
        self.assertGreaterEqual(count, 130)
        self.assertLessEqual(count, 140)
        self.assertTrue(corrected.narration.startswith(opening))
        self.assertTrue(corrected.narration.endswith(closing))
        self.assertIn("42 percent", corrected.narration)
        self.assertIn("128000-token", corrected.narration)

    def test_repair_prompt_matches_v51_generation_target(self) -> None:
        prompt = _augment_repair_prompt("BASE")

        self.assertIn("130-134", prompt)
        self.assertIn("publication validator still allows 130-140", prompt)
        self.assertNotIn("132-138", prompt)
        self.assertIn("at most 18 words", prompt)
        self.assertIn("Preserve source URLs", prompt)


if __name__ == "__main__":
    unittest.main()
