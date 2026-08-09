from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_vimax_editorial_grammar_v58 import (
    apply_editorial_visual_grammar_v58,
    classify_editorial_domain_v58,
)
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ViMaxEditorialGrammarV58Tests(unittest.TestCase):
    @staticmethod
    def _gaming_package():
        bodies = (
            "GeForce NOW adds 26 new games this August.",
            "GeForce NOW hosts a gaming event at QuakeCon.",
            "Eight new titles are now available for members.",
            "The platform is expanding its game selection.",
            "The platform is a key part of NVIDIA's gaming ecosystem.",
            "The event is a major milestone for the platform.",
        )
        return SimpleNamespace(
            topic="GeForce NOW Adds 26 New Games in August",
            title="GeForce NOW Adds 26 New Games in August",
            narration="Gaming event at QuakeCon with new games and cloud gaming access.",
            scenes=tuple(
                SimpleNamespace(
                    heading=f"Beat {index}",
                    body=body,
                    visual="A screen showing game information",
                )
                for index, body in enumerate(bodies)
            ),
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
                    f"[VIMAX_SHOT_INDEX={index}] Factual technology documentary shot synchronized to this exact spoken sentence: "
                    "The platform is expanding. Supporting source-grounded visual direction: a developer integration bench with an inference appliance. "
                    "Shot treatment: medium eye-level documentary framing. ViMax first frame: generic server rack."
                ),
                motion_prompt="Static camera. No significant changes.",
                negative_prompt="text, logo",
                continuity_anchor="same subject",
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=index + 1,
                duration_seconds=2.9,
            )
            for index in range(20)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="preserve subjects",
            image_model="image",
            video_model="video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="abc",
            scenes=scenes,
        )

    def test_geforce_story_is_classified_as_gaming(self) -> None:
        self.assertEqual("gaming", classify_editorial_domain_v58(self._gaming_package()))

    def test_gaming_story_replaces_generic_inference_lab_broll(self) -> None:
        updated = apply_editorial_visual_grammar_v58(self._plan(), self._gaming_package())
        self.assertEqual(20, len(updated.scenes))
        self.assertTrue(all(scene.generation_mode == "wan_i2v" for scene in updated.scenes))

        first = updated.scenes[0].image_prompt.casefold()
        event = updated.scenes[4].image_prompt.casefold()
        new_titles = updated.scenes[7].image_prompt.casefold()
        self.assertIn("gamer", first)
        self.assertIn("controller", first)
        self.assertIn("gaming convention", event)
        self.assertIn("game", new_titles)

        for scene in updated.scenes:
            lowered = scene.image_prompt.casefold()
            self.assertNotIn("model-serving", lowered)
            self.assertNotIn("inference appliance", lowered)
            self.assertNotIn("developer integration bench", lowered)
            self.assertNotIn("static camera", scene.motion_prompt.casefold())
            self.assertIn("native temporal-generation requirement", scene.motion_prompt.casefold())

    def test_non_gaming_ai_story_keeps_ai_software_grammar(self) -> None:
        package = SimpleNamespace(
            topic="New inference API for open models",
            title="Inference API release",
            narration="Developers can route model inference through a new API.",
            scenes=(SimpleNamespace(heading="API", body="Developers can route model inference through a new API.", visual="API workflow"),),
        )
        self.assertEqual("ai_software", classify_editorial_domain_v58(package))


if __name__ == "__main__":
    unittest.main()
