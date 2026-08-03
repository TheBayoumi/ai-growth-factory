from __future__ import annotations

import unittest

from factory.models import Scene, VideoPackage
from factory.production_narration_length import stabilize_video_package


class ProductionPackageNarrationBoundaryTests(unittest.TestCase):
    @staticmethod
    def _narration(count: int) -> str:
        closing = "Follow for the next verified production result today."
        prefix_count = count - len(closing.split())
        return " ".join(f"evidence{index}" for index in range(prefix_count)) + " " + closing

    def _package(self, narration: str) -> VideoPackage:
        return VideoPackage(
            topic="Orchard agent framework",
            narration=narration,
            title="Microsoft Research explains the Orchard agent framework",
            description="Official source explanation.",
            tags=["Orchard", "agents", "AI", "framework", "research", "tools", "systems", "workflow"],
            thumbnail_text="ORCHARD EXPLAINED",
            top_comment="What would you test first?",
            scenes=tuple(
                Scene(
                    heading=f"Scene {index}",
                    body=f"Official evidence point {index} from Orchard.",
                    visual="physical agent workflow",
                    source_index=0,
                )
                for index in range(6)
            ),
            source_urls=("https://example.com/orchard",),
            source_publishers=("Microsoft Research",),
        )

    def test_125_word_parsed_package_reaches_editorial_range(self) -> None:
        original = self._package(self._narration(125))

        corrected = stabilize_video_package(original)

        count = len(corrected.narration.split())
        self.assertGreaterEqual(count, 135)
        self.assertLessEqual(count, 155)
        self.assertTrue(
            corrected.narration.endswith(
                "Follow for the next verified production result today."
            )
        )
        self.assertIn("open the linked primary sources", corrected.narration)
        self.assertEqual(len(original.narration.split()), 125)

    def test_valid_parsed_package_is_preserved(self) -> None:
        original = self._package(self._narration(140))
        self.assertIs(stabilize_video_package(original), original)

    def test_very_short_parsed_package_remains_fail_closed(self) -> None:
        original = self._package(self._narration(79))
        self.assertIs(stabilize_video_package(original), original)


if __name__ == "__main__":
    unittest.main()
