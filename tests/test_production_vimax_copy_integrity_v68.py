from __future__ import annotations

import unittest

from factory.models import Scene, VideoPackage
from factory.production_vimax_copy_integrity_v68 import (
    normalize_finished_copy_v68,
    preserve_finished_copy_v68,
)


class ProductionViMaxCopyIntegrityV68Tests(unittest.TestCase):
    @staticmethod
    def _package() -> VideoPackage:
        return VideoPackage(
            topic="topic",
            narration="Old narration.",
            title="Valid Editorial Title For Testing",
            description="Description.",
            tags=["one", "two", "three", "four", "five", "six", "seven", "eight"],
            thumbnail_text="THUMBNAIL",
            top_comment="Comment?",
            scenes=[
                Scene(
                    heading=f"Beat {index}",
                    body=f"Old body {index}.",
                    visual=f"Immutable visual {index}.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/source"],
            source_publishers=["Publisher"],
        )

    def test_finished_copy_normalization_preserves_terminal_punctuation(self) -> None:
        self.assertEqual("Sentence ends here.", normalize_finished_copy_v68("  Sentence   ends here.  "))
        self.assertEqual("Question?", normalize_finished_copy_v68(" Question? "))

    def test_wrapper_restores_narration_and_scene_body_punctuation_only(self) -> None:
        package = self._package()
        raw = {
            "title": "Rewritten title.",
            "narration": "Rewritten narration keeps its final period.",
            "scenes": [
                {"scene_id": index, "heading": f"New Beat {index}.", "body": f"Finished body {index}."}
                for index in range(6)
            ],
        }

        def strict_base(before: VideoPackage, _raw: dict[str, object]) -> VideoPackage:
            # Simulate v66's strict fragment cleaner: title/heading/body/narration terminal
            # punctuation is stripped before the v68 finished-copy restoration layer.
            return VideoPackage(
                topic=before.topic,
                narration="Rewritten narration keeps its final period",
                title="Rewritten title",
                description=before.description,
                tags=before.tags,
                thumbnail_text=before.thumbnail_text,
                top_comment=before.top_comment,
                scenes=[
                    Scene(
                        heading=f"New Beat {index}",
                        body=f"Finished body {index}",
                        visual=scene.visual,
                        source_index=scene.source_index,
                    )
                    for index, scene in enumerate(before.scenes)
                ],
                source_urls=before.source_urls,
                source_publishers=before.source_publishers,
            )

        restored = preserve_finished_copy_v68(package, raw, base_apply=strict_base)

        self.assertEqual("Rewritten narration keeps its final period.", restored.narration)
        self.assertTrue(all(scene.body.endswith(".") for scene in restored.scenes))
        self.assertEqual("Rewritten title", restored.title)
        self.assertTrue(all(not scene.heading.endswith(".") for scene in restored.scenes))
        self.assertEqual(package.source_urls, restored.source_urls)
        self.assertEqual(package.source_publishers, restored.source_publishers)
        self.assertEqual(
            [scene.visual for scene in package.scenes],
            [scene.visual for scene in restored.scenes],
        )


if __name__ == "__main__":
    unittest.main()
