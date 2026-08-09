from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.models import Scene, VideoPackage
from factory.production_narration_integrity_v28 import validate_narration_integrity_v28


class NarrationIntegrityV28Tests(unittest.TestCase):
    def _package(self, narration: str) -> VideoPackage:
        return VideoPackage(
            topic="Orchard",
            narration=narration,
            title="Microsoft Launches Orchard Framework",
            description="description",
            tags=["ai"],
            thumbnail_text="Orchard Framework",
            top_comment="comment",
            scenes=[Scene("Orchard", "A factual claim", "A research workspace", 0)] * 6,
            source_urls=["https://example.com/orchard"],
            source_publishers=["Microsoft Research"],
        )

    def _sources(self):
        return [
            SimpleNamespace(
                url="https://example.com/orchard",
                title="Orchard: An open framework for scalable agentic AI",
            )
        ]

    def test_rejects_the_exact_failed_canary_corruption(self) -> None:
        narration = (
            "The framework is available for the research community to use and build upon, "
            "fostering continued Orchard An open framework for scalable agentic research."
        )
        with self.assertRaisesRegex(Exception, "malformed transition|source title pasted"):
            validate_narration_integrity_v28(self._package(narration), self._sources())

    def test_allows_a_grammatical_source_title_reference(self) -> None:
        narration = (
            "Microsoft Research released Orchard. The article is titled Orchard: An open "
            "framework for scalable agentic AI. Researchers can inspect the linked source."
        )
        validate_narration_integrity_v28(self._package(narration), self._sources())

    def test_allows_normal_product_mentions(self) -> None:
        narration = (
            "Microsoft Research released Orchard as an open framework. It supports agent "
            "training and evaluation while keeping infrastructure reusable."
        )
        validate_narration_integrity_v28(self._package(narration), self._sources())


if __name__ == "__main__":
    unittest.main()
