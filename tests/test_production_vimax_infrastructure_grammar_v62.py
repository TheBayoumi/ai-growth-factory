from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_vimax_infrastructure_grammar_v62 import (
    apply_ai_infrastructure_grammar_v62,
    is_ai_infrastructure_story_v62,
)
from factory.production_vimax_visual_authority_v52 import compile_vimax_image_prompt_v52
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ViMaxInfrastructureGrammarV62Tests(unittest.TestCase):
    @staticmethod
    def _package() -> SimpleNamespace:
        scenes = [
            SimpleNamespace(heading="AI Factory Launch", body="Firebird launches largest AI factory in CIS region in Armenia"),
            SimpleNamespace(heading="Accelerated Compute", body="Uses NVIDIA's accelerated computing and Dell's infrastructure"),
            SimpleNamespace(heading="Regional Research", body="Aims to boost AI research and development in the region"),
            SimpleNamespace(heading="Infrastructure Scale", body="Marks a significant step in global AI infrastructure growth"),
            SimpleNamespace(heading="Collaboration", body="Focus on advancing AI technologies and fostering collaboration"),
            SimpleNamespace(heading="Global Expansion", body="Reflects a broader trend in AI infrastructure expansion"),
        ]
        return SimpleNamespace(
            topic="AI Infrastructure Expansion",
            title="Firebird Launches AI Factory in Armenia",
            narration=(
                "NVIDIA's AI cloud Firebird launched a large AI factory using accelerated computing, "
                "dense server infrastructure, cooling, and high-performance compute capacity in Armenia."
            ),
            scenes=scenes,
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
                    "Supporting source-grounded visual direction: generic developer at a workstation. "
                    "Shot treatment: wide eye-level documentary framing. "
                    "ViMax first frame: generic developer at a workstation."
                ),
                motion_prompt="static camera",
                negative_prompt="text, logo, watermark",
                continuity_anchor="generic",
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=index + 100,
                duration_seconds=2.9,
            )
            for index in range(20)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="photorealistic technology documentary",
            palette="neutral graphite and cool practical light",
            lighting="natural technical documentary lighting",
            continuity_bible="same unbranded facility and technical crew",
            image_model="test-image-model",
            video_model="test-video-model",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="a" * 64,
            scenes=scenes,
        )

    def test_current_firebird_story_is_classified_as_ai_infrastructure(self) -> None:
        self.assertTrue(is_ai_infrastructure_story_v62(self._package()))

    def test_generation_direction_does_not_leak_ambiguous_company_name(self) -> None:
        package = self._package()
        plan = apply_ai_infrastructure_grammar_v62(self._plan(), package)
        scene = plan.scenes[0]
        self.assertIn("Firebird launches", scene.image_prompt)
        compiled = compile_vimax_image_prompt_v52(scene.image_prompt)
        lowered = compiled.compiled_prompt.casefold()
        self.assertNotIn("firebird", lowered)
        self.assertNotIn("superhero", lowered)
        self.assertNotIn("mythical", lowered)
        self.assertTrue("data-center" in lowered or "compute" in lowered or "server" in lowered)

    def test_twenty_shots_use_facility_hardware_research_and_collaboration_grammar(self) -> None:
        plan = apply_ai_infrastructure_grammar_v62(self._plan(), self._package())
        prompts = [scene.image_prompt.casefold() for scene in plan.scenes]
        joined = " ".join(prompts)
        self.assertIn("data-center campus", joined)
        self.assertIn("accelerator server", joined)
        self.assertIn("research engineers", joined)
        self.assertIn("generic officials", joined)
        self.assertIn("cooling", joined)
        directions = [
            value.split("supporting source-grounded visual direction:", 1)[1].split("shot treatment:", 1)[0].strip()
            for value in prompts
        ]
        self.assertGreaterEqual(len(set(directions)), 18)
        self.assertTrue(all("visually support this factual claim" not in value for value in directions))

    def test_generic_ai_software_without_physical_infrastructure_does_not_trigger(self) -> None:
        package = SimpleNamespace(
            topic="AI coding assistant",
            title="New agent API",
            narration="A software agent can call tools through an API and help developers write code.",
            scenes=[SimpleNamespace(body="Developers use a software agent", visual="developer workstation")],
        )
        self.assertFalse(is_ai_infrastructure_story_v62(package))

    def test_downstream_data_center_visual_hint_does_not_reclassify_ai_adoption_story(self) -> None:
        package = SimpleNamespace(
            topic="AI adoption in tax advisory",
            title="HSP GRUPPE adopts AI for tax advisory",
            narration=(
                "HSP GRUPPE integrated ChatGPT Enterprise to improve advisory productivity, client service, "
                "continuous learning, and governed professional workflows."
            ),
            scenes=[
                SimpleNamespace(
                    heading="Security Focus",
                    body="AI use is governed by strict data protection and confidentiality policies.",
                    visual="A secure data center with security protocols in place",
                ),
                SimpleNamespace(
                    heading="Weekly Use",
                    body="Employees use ChatGPT weekly for legal research and financial analysis.",
                    visual="developer workstation",
                ),
            ],
        )
        self.assertFalse(is_ai_infrastructure_story_v62(package))


if __name__ == "__main__":
    unittest.main()
