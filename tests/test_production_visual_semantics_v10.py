import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
