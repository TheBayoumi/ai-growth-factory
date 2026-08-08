from __future__ import annotations

import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_content import _validate_evidence_specificity, _validate_release_authority
from factory.production_editorial_boundary_v65 import (
    normalize_final_raw_package_v65,
    repair_hosting_publisher_attribution_v65,
    stabilize_final_package_v65,
)


class ProductionEditorialBoundaryV65Tests(unittest.TestCase):
    @staticmethod
    def _source(
        *,
        publisher: str = "NVIDIA",
        author: str = "Rev Lebaredian",
        title: str = "Firebird Launches CIS Region's Largest AI Factory in Armenia",
        summary: str | None = None,
    ) -> SourceItem:
        return SourceItem(
            publisher=publisher,
            title=title,
            url="https://example.com/firebird",
            summary=summary or (
                "Firebird launched a new AI factory in Armenia with expanded compute infrastructure, "
                "data-center capacity, cooling, power delivery, engineering operations, and regional "
                "research access. The supplied article describes deployment, commissioning, expected "
                "workloads, infrastructure expansion, and practical implementation details for local teams."
            ),
            published_at=datetime.now(timezone.utc),
            author=author,
        )

    @staticmethod
    def _package(*, narration: str, title: str, body: str = "Concrete source-backed infrastructure evidence.") -> VideoPackage:
        return VideoPackage(
            topic="Firebird AI factory",
            narration=narration,
            title=title,
            description="Source-backed infrastructure report.",
            tags=["AI", "infrastructure", "compute", "Armenia", "data center", "engineering", "GPU", "research"],
            thumbnail_text="FIREBIRD AI FACTORY",
            top_comment="What would you verify first?",
            scenes=[
                Scene(
                    heading="Infrastructure expansion",
                    body=body,
                    visual="Technicians commissioning new compute racks.",
                    source_index=0,
                )
                for _ in range(6)
            ],
            source_urls=["https://example.com/firebird"],
            source_publishers=["NVIDIA"],
        )

    def test_final_raw_boundary_repairs_31_word_scene_without_metadata_changes(self) -> None:
        body = " ".join(f"word{index}" for index in range(31))
        raw = {
            "topic": "topic",
            "narration": "narration",
            "title": "title",
            "description": "description",
            "source_urls": ["https://example.com/firebird"],
            "source_publishers": ["NVIDIA"],
            "scenes": [
                {
                    "heading": "One two three four five six seven",
                    "body": body,
                    "visual": "visual",
                    "source_index": 0,
                }
            ],
        }
        corrected = normalize_final_raw_package_v65(raw, [self._source()])
        scene = corrected["scenes"][0]
        self.assertEqual(18, len(scene["body"].split()))
        self.assertLessEqual(len(scene["heading"].split()), 5)
        self.assertEqual(0, scene["source_index"])
        self.assertEqual(raw["source_urls"], corrected["source_urls"])
        self.assertEqual(raw["source_publishers"], corrected["source_publishers"])

    def test_gross_scene_overflow_remains_for_strict_validator(self) -> None:
        body = " ".join(f"word{index}" for index in range(48))
        raw = {"scenes": [{"heading": "heading", "body": body, "visual": "visual", "source_index": 0}]}
        corrected = normalize_final_raw_package_v65(raw, [self._source()])
        self.assertEqual(48, len(corrected["scenes"][0]["body"].split()))

    def test_hosting_publisher_actor_is_replaced_only_from_source_title(self) -> None:
        source = self._source()
        package = self._package(
            narration="NVIDIA launched the region's largest AI factory, expanding local compute capacity. " + "Evidence remains source backed. " * 18,
            title="NVIDIA Launches Major AI Factory Expansion",
        )
        corrected = repair_hosting_publisher_attribution_v65(package, [source])
        self.assertTrue(corrected.title.startswith("Firebird Launches"))
        self.assertTrue(corrected.narration.startswith("Firebird launched"))
        _validate_release_authority(corrected, [source])

    def test_no_title_actor_means_no_attribution_guess(self) -> None:
        source = self._source(title="Inside Armenia's New AI Infrastructure Program")
        package = self._package(
            narration="NVIDIA launched a major new AI infrastructure program. " + "Evidence remains source backed. " * 18,
            title="NVIDIA Launches AI Infrastructure Program",
        )
        corrected = repair_hosting_publisher_attribution_v65(package, [source])
        self.assertEqual(package.title, corrected.title)
        self.assertEqual(package.narration, corrected.narration)
        with self.assertRaisesRegex(Exception, "hosting publisher"):
            _validate_release_authority(corrected, [source])

    def test_thin_evidence_gate_is_not_bypassed(self) -> None:
        thin = self._source(summary="Too little primary evidence to support four concrete facts.")
        package = self._package(
            narration="Firebird launched a new AI factory in Armenia. " + "Evidence remains source backed. " * 18,
            title="Firebird Launches Major AI Factory Expansion",
        )
        corrected = stabilize_final_package_v65(package, [thin])
        with self.assertRaisesRegex(Exception, "too thin"):
            _validate_evidence_specificity(corrected, [thin])


if __name__ == "__main__":
    unittest.main()
