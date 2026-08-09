from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_vimax_human_editorial_v66 import (
    _AI_INFRA_SHOTS,
    _repair_scene_actor,
    apply_human_editorial_storyboard_v66,
    consumer_editorial_failures_v66,
    visual_family_counts_v66,
)


@dataclass(frozen=True)
class FakeVisualScene:
    scene_index: int
    image_prompt: str = "old"
    motion_prompt: str = "old"
    continuity_anchor: str = ""


@dataclass(frozen=True)
class FakePlan:
    prompt_version: str
    scenes: tuple[FakeVisualScene, ...]


class ProductionViMaxHumanEditorialV66Tests(unittest.TestCase):
    @staticmethod
    def _source() -> SourceItem:
        return SourceItem(
            publisher="NVIDIA",
            author="Rev Lebaredian",
            title="Firebird Launches CIS Region's Largest AI Factory in Armenia",
            url="https://example.com/firebird",
            summary=(
                "Firebird launched a large AI factory in Armenia using accelerated compute and high-performance infrastructure. "
                "The facility is intended to support local AI research, engineering workloads, infrastructure capacity, and practical deployment."
            ),
            published_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _package(narration: str) -> VideoPackage:
        return VideoPackage(
            topic="Firebird launches AI factory in Armenia",
            narration=narration,
            title="Firebird Launches Major AI Factory",
            description="Source-backed report.",
            tags=["AI", "Armenia", "compute", "infrastructure", "engineering", "data center", "research", "deployment"],
            thumbnail_text="AI FACTORY",
            top_comment="What matters most?",
            scenes=[
                Scene(
                    heading=f"Beat {index}",
                    body=(
                        "NVIDIA launches CIS region's largest AI factory in Armenia."
                        if index == 0
                        else "The facility expands practical AI compute infrastructure for local engineering workloads."
                    ),
                    visual="Physical infrastructure documentary scene.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/firebird"],
            source_publishers=["NVIDIA"],
        )

    def test_internal_provenance_language_is_rejected(self) -> None:
        package = self._package(
            "Firebird launched a new AI factory in Armenia. NVIDIA provide separate primary-source context for this topic. "
            "Each report is evaluated independently and supports only its attributed claim. "
            + "Concrete infrastructure evidence remains in the supplied source. " * 15
        )
        failures = consumer_editorial_failures_v66(package)
        self.assertTrue(any("internal source/provenance" in item for item in failures))

    def test_internal_provenance_in_scene_copy_is_rejected(self) -> None:
        package = self._package(
            "Firebird launched a new AI factory in Armenia. "
            + "Concrete infrastructure evidence remains in the supplied source. " * 20
        )
        scenes = list(package.scenes)
        scenes[3] = Scene(
            heading="Independent source context",
            body="The selected reports are evaluated separately and support only their attributed claims.",
            visual=scenes[3].visual,
            source_index=scenes[3].source_index,
        )
        package = VideoPackage(
            topic=package.topic,
            narration=package.narration,
            title=package.title,
            description=package.description,
            tags=package.tags,
            thumbnail_text=package.thumbnail_text,
            top_comment=package.top_comment,
            scenes=scenes,
            source_urls=package.source_urls,
            source_publishers=package.source_publishers,
        )
        failures = consumer_editorial_failures_v66(package)
        self.assertTrue(any("scene copy" in item for item in failures))

    def test_multiple_generic_filler_claims_are_rejected(self) -> None:
        package = self._package(
            "Firebird launched a new AI facility. This is part of a broader trend in AI infrastructure growth. "
            "The facility supports a wide range of AI applications and is a key step in expanding AI capabilities. "
            + "The supplied source contains concrete deployment details. " * 15
        )
        failures = consumer_editorial_failures_v66(package)
        self.assertTrue(any("generic filler" in item for item in failures))

    def test_concrete_consumer_narration_passes_copy_gate(self) -> None:
        package = self._package(
            "Firebird launched a new AI factory in Armenia using accelerated computing and high-performance infrastructure. "
            "The facility adds dedicated compute capacity for local engineering and research workloads. "
            + "Technicians can commission dense compute, cooling, networking, and power infrastructure for controlled AI workloads. " * 9
        )
        self.assertEqual((), consumer_editorial_failures_v66(package))

    def test_scene_actor_uses_source_title_actor_not_host_publisher(self) -> None:
        package = self._package("Firebird launched a new AI factory in Armenia. " + "Concrete evidence follows. " * 30)
        repaired = _repair_scene_actor(package, [self._source()])
        self.assertTrue(repaired.scenes[0].body.startswith("Firebird launches"))
        self.assertNotIn("NVIDIA launches", repaired.scenes[0].body)

    def test_storyboard_is_twenty_simple_continuous_actions(self) -> None:
        package = self._package("Firebird launched a new AI factory in Armenia. " + "Concrete infrastructure evidence follows. " * 30)
        plan = FakePlan(
            prompt_version="vimax-script2video@test",
            scenes=tuple(FakeVisualScene(index) for index in range(20)),
        )
        updated = apply_human_editorial_storyboard_v66(plan, package)
        self.assertEqual(20, len(updated.scenes))
        self.assertEqual(20, len(_AI_INFRA_SHOTS))
        for scene in updated.scenes:
            lowered = scene.image_prompt.casefold()
            self.assertIn("single continuous", lowered)
            self.assertNotIn("humanoid robot", lowered)
            self.assertNotIn("articulated robot", lowered)
            self.assertIn("native temporal-generation requirement", scene.motion_prompt.casefold())

    def test_opening_and_full_sequence_have_editorial_family_diversity(self) -> None:
        counts = visual_family_counts_v66([SimpleNamespace() for _ in range(20)])
        self.assertGreaterEqual(len(counts), 16)
        first_four = [item[0] for item in _AI_INFRA_SHOTS[:4]]
        self.assertEqual(4, len(set(first_four)))


if __name__ == "__main__":
    unittest.main()
