from __future__ import annotations

import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_story_coherence import (
    StoryCoherenceError,
    repair_thumbnail_copy,
    validate_story_coherence,
)


def _source(publisher: str, title: str, url: str, summary: str) -> SourceItem:
    return SourceItem(
        publisher=publisher,
        title=title,
        url=url,
        summary=summary,
        published_at=datetime.now(timezone.utc),
    )


def _package(urls: list[str], publishers: list[str], scenes: list[Scene]) -> VideoPackage:
    return VideoPackage(
        topic="EvoLib adaptive knowledge",
        narration=(
            "Microsoft Research introduced EvoLib as a method for converting repeated "
            "experience into reusable knowledge. The system captures lessons from completed "
            "tasks, organizes them, and applies those lessons to later work. A separate "
            "evaluation report examines EvoLib behavior under controlled tasks and measures "
            "whether retained knowledge improves later decisions. Engineers should compare "
            "accuracy, latency, corrections, and repeatability before changing a production "
            "workflow. The important question is not whether the demo looks intelligent, "
            "but whether the same stored lesson improves a new task without introducing "
            "unsupported behavior. That makes EvoLib a concrete test of persistent agent "
            "learning rather than a broad claim about every AI system. Follow for the next "
            "source-verified result and the measurements that determine whether it works."
        ),
        title="Microsoft Research: EvoLib Explained",
        description="Source-grounded EvoLib report.",
        tags=["EvoLib", "AI agents"],
        thumbnail_text="EvoLib Turning experience into",
        top_comment="Which EvoLib metric matters most?",
        scenes=scenes,
        source_urls=urls,
        source_publishers=publishers,
    )


class ProductionStoryCoherenceTests(unittest.TestCase):
    def test_rejects_unrelated_secondary_source(self) -> None:
        primary = _source(
            "Microsoft Research",
            "EvoLib: Turning experience into evolving knowledge",
            "https://example.com/evolib",
            "EvoLib stores lessons from completed tasks for later agent decisions.",
        )
        unrelated = _source(
            "Google AI",
            "Expanding managed agents with Gemini API hooks",
            "https://example.com/gemini-hooks",
            "Gemini API hooks simplify managed tool execution.",
        )
        package = _package(
            [primary.url, unrelated.url],
            [primary.publisher, unrelated.publisher],
            [Scene("EvoLib", "EvoLib stores reusable experience.", "physical memory", 0)] * 6,
        )

        with self.assertRaisesRegex(StoryCoherenceError, "unrelated to the primary story"):
            validate_story_coherence(package, [primary, unrelated])

    def test_rejects_primary_entity_borrowed_by_secondary_scene(self) -> None:
        primary = _source(
            "Microsoft Research",
            "EvoLib: Turning experience into evolving knowledge",
            "https://example.com/evolib",
            "EvoLib stores lessons from completed tasks.",
        )
        related = _source(
            "Independent Lab",
            "Evolving knowledge evaluation under repeated tasks",
            "https://example.com/evolving-knowledge-eval",
            "The evaluation measures retained task knowledge and repeated decisions.",
        )
        scenes = [Scene("EvoLib", "EvoLib stores reusable experience.", "memory", 0)] * 5
        scenes.append(
            Scene(
                "Wrong attribution",
                "EvoLib is presented as the evaluated system in this unrelated source.",
                "generic portrait",
                1,
            )
        )
        package = _package(
            [primary.url, related.url],
            [primary.publisher, related.publisher],
            scenes,
        )

        with self.assertRaisesRegex(StoryCoherenceError, "primary subject token"):
            validate_story_coherence(package, [primary, related])

    def test_accepts_related_sources_and_repairs_complete_thumbnail(self) -> None:
        primary = _source(
            "Microsoft Research",
            "EvoLib: Turning experience into evolving knowledge",
            "https://example.com/evolib",
            "EvoLib stores lessons from completed tasks.",
        )
        related = _source(
            "Independent Lab",
            "EvoLib evaluation under repeated tasks",
            "https://example.com/evolib-eval",
            "EvoLib evaluation measures retained task knowledge.",
        )
        scenes = [
            Scene("EvoLib memory", "EvoLib stores reusable task experience.", "memory", 0),
            Scene("EvoLib test", "EvoLib evaluation measures repeated tasks.", "test", 1),
            Scene("Mechanism", "Stored lessons guide later decisions.", "pathway", 0),
            Scene("Evidence", "Repeated tasks measure retained knowledge.", "comparison", 1),
            Scene("Implication", "Teams compare corrections and repeatability.", "workspace", 1),
            Scene("Decision", "Engineers verify results before deployment.", "choice", 0),
        ]
        package = _package(
            [primary.url, related.url],
            [primary.publisher, related.publisher],
            scenes,
        )

        validate_story_coherence(package, [primary, related])
        repaired = repair_thumbnail_copy(package, [primary, related])
        self.assertEqual(repaired.thumbnail_text, "EVOLIB EXPLAINED")


if __name__ == "__main__":
    unittest.main()
