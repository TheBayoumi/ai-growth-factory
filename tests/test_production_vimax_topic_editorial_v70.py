from __future__ import annotations

import unittest

from factory.models import Scene, VideoPackage
from factory.production_vimax_topic_editorial_v70 import (
    apply_topic_editorial_storyboard_v70,
    is_ai_infrastructure_story_v70,
    is_professional_services_story_v70,
    story_world_v70,
    validate_topic_editorial_diversity_v70,
)
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ProductionViMaxTopicEditorialV70Tests(unittest.TestCase):
    @staticmethod
    def _hsp_package() -> VideoPackage:
        return VideoPackage(
            topic="HSP GRUPPE adopts AI for tax advisory",
            narration=(
                "A network of tax and legal firms integrated an enterprise AI assistant to improve productivity and client service. "
                "Employees use it for advisory work, legal research, governance, continuous learning, and knowledge sharing."
            ),
            title="HSP GRUPPE adopts AI for tax advisory",
            description="Source-backed report.",
            tags=["AI", "tax", "legal", "advisory", "productivity", "governance", "enterprise", "professional services"],
            thumbnail_text="AI TAX ADVISORY",
            top_comment="What would you verify?",
            scenes=[
                Scene(
                    heading=f"Beat {index}",
                    body=(
                        "Employees report higher productivity and more capacity for advisory work."
                        if index < 3
                        else "The firm uses governance, learning forums, and data protection for responsible adoption."
                    ),
                    # Reproduce the live bug: this non-authoritative suggestion must not control story routing.
                    visual="A secure data center with security protocols in place.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/hsp"],
            source_publishers=["OpenAI"],
        )

    @staticmethod
    def _firebird_package() -> VideoPackage:
        base = ProductionViMaxTopicEditorialV70Tests._hsp_package()
        return VideoPackage(
            topic="Firebird launches an AI factory in Armenia",
            narration="Firebird launched an AI factory with accelerated computing, new compute capacity, cooling, and facility infrastructure.",
            title="Firebird launches major AI factory",
            description=base.description,
            tags=base.tags,
            thumbnail_text="AI FACTORY",
            top_comment=base.top_comment,
            scenes=[
                Scene("AI factory", "The AI factory expands regional compute capacity.", "generic", 0)
                for _ in range(6)
            ],
            source_urls=base.source_urls,
            source_publishers=base.source_publishers,
        )

    @staticmethod
    def _plan() -> VisualPlan:
        scenes = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=0,
                role="hook" if index == 0 else "evidence",
                generation_mode="wan_i2v",
                image_prompt=f"[VIMAX_SHOT_INDEX={index}] old generic visual {index}",
                motion_prompt="static camera",
                negative_prompt="text, logo",
                continuity_anchor="old",
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=1000 + index,
                duration_seconds=2.8,
            )
            for index in range(20)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="continuous",
            image_model="image",
            video_model="video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="a" * 64,
            scenes=scenes,
        )

    def test_generated_scene_visual_cannot_misroute_professional_services_story(self) -> None:
        package = self._hsp_package()
        self.assertFalse(is_ai_infrastructure_story_v70(package))
        self.assertTrue(is_professional_services_story_v70(package))
        self.assertEqual("professional_services", story_world_v70(package))

    def test_factual_ai_factory_story_still_routes_to_infrastructure(self) -> None:
        package = self._firebird_package()
        self.assertTrue(is_ai_infrastructure_story_v70(package))
        self.assertEqual("ai_infrastructure", story_world_v70(package))

    def test_professional_services_storyboard_preserves_slots_but_replaces_wrong_world(self) -> None:
        updated = apply_topic_editorial_storyboard_v70(self._plan(), self._hsp_package())
        self.assertEqual(20, len(updated.scenes))
        self.assertEqual([2.8] * 20, [scene.duration_seconds for scene in updated.scenes])
        self.assertEqual(list(range(1000, 1020)), [scene.seed for scene in updated.scenes])
        opening = []
        for scene in updated.scenes:
            lowered = scene.image_prompt.casefold()
            self.assertIn("single continuous", lowered)
            self.assertIn("native temporal-generation requirement", scene.motion_prompt.casefold())
            self.assertNotIn("data-center", lowered)
            self.assertNotIn("server rack", lowered)
            self.assertNotIn("cooling manifold", lowered)
            self.assertNotIn("static camera", scene.motion_prompt.casefold())
            opening.append(lowered)
        self.assertEqual(4, len(set(opening[:4])))
        validate_topic_editorial_diversity_v70(updated.scenes)

    def test_generic_diversity_validator_rejects_identical_static_storyboard(self) -> None:
        plan = self._plan()
        identical = tuple(
            SceneVisualPrompt(
                scene_index=scene.scene_index,
                source_index=scene.source_index,
                role=scene.role,
                generation_mode=scene.generation_mode,
                image_prompt="Supporting source-grounded visual direction: one identical office scene.",
                motion_prompt="Static camera with no movement.",
                negative_prompt=scene.negative_prompt,
                continuity_anchor=scene.continuity_anchor,
                caption_safe_zone=scene.caption_safe_zone,
                seed=scene.seed,
                duration_seconds=scene.duration_seconds,
            )
            for scene in plan.scenes
        )
        with self.assertRaises(ValueError):
            validate_topic_editorial_diversity_v70(identical)


if __name__ == "__main__":
    unittest.main()
