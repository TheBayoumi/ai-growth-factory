import unittest
from types import SimpleNamespace

from factory.models import Scene, VideoPackage
from factory.production_visual_semantics import _scene_specific_repair
from factory.visual_prompt import SceneVisualPrompt, VisualPlan
from factory.visual_prompt_compiler import (
    compile_image_prompt,
    compile_motion_prompt,
    validate_compiled_prompt_diversity,
)


class ProductionVisualSemanticsV10Tests(unittest.TestCase):
    def _scenes(self):
        concepts = (
            "multilingual customer support represented by diverse people linked through luminous pathways",
            "a spatial arrangement with customer satisfaction rates",
            "a world map with global locations",
            "a response appearing in seconds through a short light pathway",
            "a human agent working beside an unmarked glass object with AI assistance",
            "a comparison between automated routing and human escalation paths",
        )
        roles = ("hook", "evidence", "mechanism", "comparison", "implication", "cta")
        return [
            SimpleNamespace(
                scene_index=index,
                image_prompt=(
                    "A floating abstract sphere represents the AI system. "
                    f"The scene is showing {concept}."
                ),
                negative_prompt="text, logo",
                generation_mode="wan_i2v" if index < 3 else "image",
                motion_prompt=(
                    "The abstract sphere slowly rotates while the interface glows. "
                    "The camera slowly zooms in."
                ),
                role=roles[index],
            )
            for index, concept in enumerate(concepts)
        ]

    def test_compiler_preserves_the_distinctive_scene_clause(self):
        prompts = [
            compile_image_prompt(scene.image_prompt).compiled_prompt
            for scene in self._scenes()
        ]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertIn("multilingual customer support", prompts[0])
        self.assertIn("customer satisfaction rates", prompts[1])
        self.assertIn("world map with global locations", prompts[2])
        self.assertIn("human agent", prompts[4])
        self.assertTrue(all("floating abstract sphere" not in value.lower() for value in prompts))

    def test_wan_motion_is_bound_to_each_scene_semantics_and_role(self):
        scenes = self._scenes()[:3]
        prompts = [
            compile_motion_prompt(
                scene.motion_prompt,
                semantic_context=scene.image_prompt,
                role=scene.role,
            ).compiled_motion_prompt
            for scene in scenes
        ]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertIn("multilingual customer support", prompts[0])
        self.assertIn("customer satisfaction rates", prompts[1])
        self.assertIn("world map", prompts[2])
        self.assertTrue(all("camera slowly zooms" not in value.lower() for value in prompts))

    def test_full_plan_diversity_passes_and_repeated_plan_fails(self):
        validate_compiled_prompt_diversity(self._scenes())
        repeated = self._scenes()
        for scene in repeated:
            scene.image_prompt = "A floating abstract sphere represents the AI system."
            scene.motion_prompt = "The sphere slowly rotates."
            scene.role = "hook"
        with self.assertRaisesRegex(ValueError, "not semantically distinct"):
            validate_compiled_prompt_diversity(repeated)

    def test_v10_scientist_collision_gets_deterministic_role_aware_repair(self):
        package_scenes = (
            Scene("Efficient AI Models", "GPT improves inference efficiency.", "A processor completing work quickly.", 0),
            Scene("Evolving Knowledge", "Agents adapt through experience.", "A library reorganizing itself.", 1),
            Scene("Research Transformation", "AI accelerates genomics discoveries.", "A scientist using AI to analyze genetic data.", 0),
            Scene("Productivity Boost", "Researchers focus on innovation.", "Researchers collaborating in a laboratory.", 1),
            Scene("Future of Research", "AI supports exploration and discovery.", "A wide experimental research environment.", 0),
            Scene("Innovation Access", "Tools balance speed and precision.", "A scientist using an AI tool to analyze data.", 1),
        )
        package = VideoPackage(
            topic="AI workflows",
            narration="A sufficiently descriptive narration for testing visual repair behavior.",
            title="AI Workflows Are Changing Research",
            description="Description.",
            tags=["AI", "research"],
            thumbnail_text="RESEARCH CHANGED",
            top_comment="What should be tested next?",
            scenes=list(package_scenes),
            source_urls=["https://example.com/a", "https://example.com/b"],
            source_publishers=["A", "B"],
        )
        roles = ("hook", "evidence", "mechanism", "comparison", "implication", "cta")
        prompts = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=package_scenes[index].source_index,
                role=roles[index],
                generation_mode="wan_i2v" if index < 3 else "image",
                image_prompt="A scientist using an AI tool to analyze data in a laboratory.",
                motion_prompt="The scientist moves while data glows.",
                negative_prompt="text, logo",
                continuity_anchor="blue accent",
                caption_safe_zone="lower third",
                seed=100 + index,
                duration_seconds=9.0,
            )
            for index in range(6)
        )
        plan = VisualPlan(
            prompt_version="test",
            global_style="editorial",
            palette="blue and amber",
            lighting="cinematic",
            continuity_bible="secondary accent only",
            image_model="image-model",
            video_model="video-model",
            width=704,
            height=1280,
            fps=30,
            director_input_sha256="abc",
            scenes=prompts,
        )

        repaired = _scene_specific_repair(plan, package)
        compiled = [
            compile_image_prompt(scene.image_prompt).compiled_prompt
            for scene in repaired.scenes
        ]

        self.assertEqual(len(compiled), len(set(compiled)))
        self.assertIn("genetic data", compiled[2].lower())
        self.assertIn("speed and precision", compiled[5].lower())
        self.assertIn("directional mechanism", repaired.scenes[2].image_prompt.lower())
        self.assertIn("human-scale decision", repaired.scenes[5].image_prompt.lower())
        self.assertNotEqual(repaired.scenes[2].motion_prompt, repaired.scenes[5].motion_prompt)


if __name__ == "__main__":
    unittest.main()
