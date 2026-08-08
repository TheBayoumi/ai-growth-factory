from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from factory.production_vimax_unified_storyboard_v64 import (
    _AI_INFRA_DIRECTIONS,
    apply_unified_editorial_storyboard_v64,
    validate_vimax_editorial_diversity_v64,
    visual_family_v64,
)
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ViMaxUnifiedStoryboardV64Tests(unittest.TestCase):
    @staticmethod
    def _package() -> SimpleNamespace:
        bodies = (
            "Firebird's largest AI factory in Armenia is now operational.",
            "The facility uses accelerated computing and modern infrastructure.",
            "Armenia's government supports the AI development project.",
            "The facility is designed to handle large-scale AI workloads.",
            "The project reflects a growing trend in AI infrastructure expansion.",
            "Technology partners collaborate on advanced AI infrastructure.",
        )
        return SimpleNamespace(
            topic="AI Infrastructure Expansion",
            title="AI Factory Launch in Armenia",
            narration="A new AI factory expands accelerated computing infrastructure and regional research capacity.",
            scenes=[SimpleNamespace(heading=f"Beat {i}", body=body, visual="") for i, body in enumerate(bodies)],
        )

    @staticmethod
    def _plan() -> VisualPlan:
        scenes = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=0,
                role="hook" if index == 0 else "evidence",
                generation_mode="wan_i2v",
                image_prompt=(
                    f"[VIMAX_SHOT_INDEX={index}] Factual technology documentary shot. "
                    "Supporting source-grounded visual direction: old generic local-AI setup. "
                    "Shot treatment: medium eye-level documentary framing. ViMax first frame: old setup."
                ),
                motion_prompt="static camera",
                negative_prompt="text, logo",
                continuity_anchor="old",
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=100 + index,
                duration_seconds=2.9,
            )
            for index in range(20)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="documentary",
            palette="graphite",
            lighting="natural",
            continuity_bible="continuous",
            image_model="image",
            video_model="video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="b" * 64,
            scenes=scenes,
        )

    def test_exact_twenty_shot_progression_is_unique_and_not_rack_only(self) -> None:
        self.assertEqual(20, len(_AI_INFRA_DIRECTIONS))
        self.assertEqual(20, len(set(_AI_INFRA_DIRECTIONS)))
        rack_count = sum("rack" in value.casefold() for value in _AI_INFRA_DIRECTIONS)
        self.assertLessEqual(rack_count, 9)
        joined = " ".join(_AI_INFRA_DIRECTIONS).casefold()
        for phrase in (
            "data-center campus",
            "loading-bay",
            "accelerator server tray",
            "fiber",
            "liquid-cooling",
            "technical delegation",
            "robotics experiment",
            "computer-vision",
            "facility-expansion exterior",
            "final wide hero",
        ):
            self.assertIn(phrase, joined)

    def test_plan_generation_and_preflight_use_same_directions(self) -> None:
        plan = apply_unified_editorial_storyboard_v64(self._plan(), self._package())
        families = validate_vimax_editorial_diversity_v64(plan.scenes)
        self.assertGreaterEqual(len(families), 8)
        self.assertLessEqual(max(families.values()), 5)
        for index, scene in enumerate(plan.scenes):
            self.assertIn(_AI_INFRA_DIRECTIONS[index], scene.image_prompt)
            self.assertNotIn("old generic local-AI setup", scene.image_prompt)
            self.assertEqual("wan_i2v", scene.generation_mode)

    def test_family_distribution_contains_editorial_progression(self) -> None:
        families = Counter(visual_family_v64(value) for value in _AI_INFRA_DIRECTIONS)
        for expected in (
            "facility_exterior",
            "logistics_commissioning",
            "accelerator_hardware",
            "network_fiber",
            "cooling_thermal",
            "power_distribution",
            "delegation_briefing",
            "research_robotics",
            "research_vision",
            "application_validation",
            "cluster_scale",
        ):
            self.assertIn(expected, families)

    def test_non_infrastructure_plan_is_not_rewritten(self) -> None:
        package = SimpleNamespace(
            topic="Coding agent API",
            title="New software agent",
            narration="A coding agent can call tools through an API.",
            scenes=[SimpleNamespace(heading="API", body="Developers call tools", visual="software")],
        )
        plan = self._plan()
        self.assertIs(plan, apply_unified_editorial_storyboard_v64(plan, package))


if __name__ == "__main__":
    unittest.main()
