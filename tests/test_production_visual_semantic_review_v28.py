from __future__ import annotations

import unittest
from pathlib import Path

from factory.production_visual_convergence_v29 import (
    SDXLQualityKeyframeGeneratorV29,
    _concept_for,
    _physical_repair,
    _scene_for_attempt_v29,
)
from factory.production_visual_prompt_cleanup_v29 import (
    compile_display_free_physical_prompt_v29,
)
from factory.production_visual_semantic_review_v28 import (
    _base_director_prompt,
    _extract_direction,
    _normalize_visual_intent,
    _sanitize_negative_prompt,
    _sanitize_repair,
    _shot_setup,
    compile_semantic_generation_prompt_v28,
)
from factory.visual_prompt import SceneVisualPrompt


FAILED_CANARY_PROMPT = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "Microsoft Research has launched a new open-source framework for training and evaluating "
    "AI agents, designed to simplify research while maintaining strong performance.. "
    "Supporting source-grounded visual direction: A researcher working on a computer with a "
    "new framework highlighted. Shot treatment: tight concrete detail with a clear foreground "
    "action. Depict the concrete idea literally. Generic unbranded researchers, workspaces, "
    "tools, devices, code-like structures, or procedural diagrams are allowed when they "
    "communicate the spoken claim. Preserve a full-frame environment."
)

FAILED_SCENE_14_PROMPT = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "The release comes as part of Microsoft's ongoing efforts to support AI research.. "
    "Supporting source-grounded visual direction: A screen showing the framework's open-source page. "
    "Shot treatment: cause-to-result process view with visible directional change. "
    "V28 SCENE SETUP: software-integration; over-shoulder workstation view. "
    "V28 REPAIR: Ensure the researcher is interacting with a screen displaying the open-source "
    "framework's page. REQUIRED CORRECTION: Ensure the display surfaces show the page content"
)

GOOGLE_COURSE_PROMPT = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "Google AI has expanded its vibe coding course to include 353,000 participants.. "
    "Supporting source-grounded visual direction: A screen showing a course expansion announcement. "
    "Shot treatment: wide contextual view showing the surrounding workflow. "
    "Depict the concrete idea literally. Preserve a full-frame environment."
)

DEVELOPER_CODING_PROMPT = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "The initiative aims to improve coding and AI skills, fostering innovation in the field.. "
    "Supporting source-grounded visual direction: A screen showing a developer coding. "
    "Shot treatment: human-scale consequence using generic unbranded people or tools when relevant."
)


class ProductionVisualSemanticReviewV28Tests(unittest.TestCase):
    def test_generation_prompt_never_receives_narration_prose(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_CANARY_PROMPT)
        self.assertNotIn("Microsoft Research has launched", compiled.compiled_prompt)
        self.assertNotIn("training and evaluating AI agents", compiled.compiled_prompt)
        self.assertNotIn("Microsoft", compiled.compiled_prompt)
        self.assertIn("researcher", compiled.compiled_prompt)
        self.assertIn("laboratory", compiled.compiled_prompt)

    def test_failed_page_direction_becomes_text_free_software_action(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_SCENE_14_PROMPT)
        lowered = compiled.compiled_prompt.casefold()
        self.assertIn("abstract geometric", lowered)
        self.assertIn("unbranded workstation", lowered)
        self.assertNotIn("framework's page", lowered)
        self.assertNotIn("page content", lowered)
        self.assertNotIn("microsoft", lowered)

    def test_v29_course_expansion_becomes_physical_workshop(self) -> None:
        compiled = compile_display_free_physical_prompt_v29(GOOGLE_COURSE_PROMPT)
        lowered = compiled.compiled_prompt.casefold()
        self.assertIn("adult developers", lowered)
        self.assertIn("physical workbench", lowered)
        self.assertIn("single camera viewpoint", lowered)
        for forbidden in ("google", "353,000", "announcement", "screen", "monitor", "display"):
            self.assertNotIn(forbidden, lowered)
        self.assertLessEqual(compiled.word_count, 62)
        self.assertEqual(
            compiled.compiler_version,
            "visual-compiler-v29-physical-story-cfg-display-free",
        )

    def test_v29_coding_story_uses_physical_programmable_hardware(self) -> None:
        compiled = compile_display_free_physical_prompt_v29(DEVELOPER_CODING_PROMPT)
        lowered = compiled.compiled_prompt.casefold()
        self.assertIn("programmable controller", lowered)
        self.assertIn("physical hardware", lowered)
        self.assertNotIn("developer coding", lowered)
        self.assertNotIn("screen", lowered)
        self.assertNotIn("laptop", lowered)
        self.assertNotIn("display", lowered)

    def test_v29_negative_prompt_can_forbid_text_collages_and_humanoids(self) -> None:
        negative = compile_display_free_physical_prompt_v29(
            GOOGLE_COURSE_PROMPT
        ).negative_prompt.casefold()
        self.assertIn("readable text", negative)
        self.assertIn("screen", negative)
        self.assertIn("collage", negative)
        self.assertIn("humanoid robot", negative)
        self.assertIn("distorted hands", negative)

    def test_prompt_budget_is_hard_and_clip_safe(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_SCENE_14_PROMPT, word_budget=500)
        self.assertLessEqual(compiled.word_count, 58)
        self.assertEqual(compiled.word_budget, 58)
        self.assertEqual(compiled.word_count, len(compiled.compiled_prompt.rstrip(".").split()))

    def test_people_hands_screens_and_devices_are_not_legacy_negative_defects(self) -> None:
        negative = _sanitize_negative_prompt(
            "people, faces, hands, screens, phones, devices, documents, panels"
        ).casefold()
        self.assertNotIn("people, faces", negative)
        self.assertNotIn("screens", negative)
        self.assertNotIn("phones", negative)
        self.assertNotIn("devices", negative)
        self.assertIn("malformed anatomy", negative)
        self.assertIn("distorted hands", negative)
        self.assertIn("readable text", negative)

    def test_graph_direction_becomes_physical_evidence_not_infographic(self) -> None:
        prompt = FAILED_CANARY_PROMPT.replace(
            "A researcher working on a computer with a new framework highlighted",
            "A graph showing performance metrics for AI models",
        )
        compiled = compile_semantic_generation_prompt_v28(prompt)
        self.assertIn("physical AI evaluation bench", compiled.compiled_prompt)
        self.assertIn("illuminated modules", compiled.compiled_prompt)
        self.assertNotIn("graph showing", compiled.compiled_prompt.casefold())
        self.assertNotIn("infographic", compiled.compiled_prompt.casefold())

    def test_repair_requesting_a_page_is_sanitized(self) -> None:
        repair = _sanitize_repair(
            "Ensure the researcher is interacting with a screen displaying the open-source framework's page"
        ).casefold()
        self.assertIn("abstract modular shapes", repair)
        self.assertIn("no readable text", repair)
        self.assertNotIn("framework's page", repair)
        self.assertLessEqual(len(repair.split()), 20)

    def test_v29_text_defect_removes_all_display_surfaces(self) -> None:
        concept = _concept_for(GOOGLE_COURSE_PROMPT, 0)
        repair = _physical_repair("The screen contains readable text and an announcement", concept)
        self.assertIn("display-free workspace", repair)
        self.assertIn("physical hardware and people only", repair)

    def test_retry_markers_are_replaced_not_accumulated(self) -> None:
        base = _base_director_prompt(FAILED_SCENE_14_PROMPT)
        self.assertNotIn("V28 SCENE SETUP", base)
        self.assertNotIn("V28 REPAIR", base)
        self.assertNotIn("REQUIRED CORRECTION", base)
        self.assertIn("Supporting source-grounded visual direction", base)

    def test_v29_retry_preserves_story_setup_and_changes_only_seed_and_repair(self) -> None:
        scene = SceneVisualPrompt(
            scene_index=4,
            source_index=0,
            role="mechanism",
            generation_mode="image",
            image_prompt=DEVELOPER_CODING_PROMPT,
            motion_prompt="controlled motion",
            negative_prompt="legacy",
            continuity_anchor="anchor",
            caption_safe_zone="lower",
            seed=123,
            duration_seconds=3.0,
        )
        first = _scene_for_attempt_v29(scene, scene_index=4, attempt=1)
        retry = _scene_for_attempt_v29(
            scene,
            scene_index=4,
            attempt=4,
            repair="The screen contains readable text",
        )
        first_setup = first.image_prompt.split("V28 SCENE SETUP:", 1)[1].split(". V28 REPAIR:", 1)[0]
        retry_setup = retry.image_prompt.split("V28 SCENE SETUP:", 1)[1].split(". V28 REPAIR:", 1)[0]
        self.assertEqual(first_setup, retry_setup)
        self.assertNotEqual(first.seed, retry.seed)
        self.assertEqual(first.image_prompt.count("V28 SCENE SETUP:"), 1)
        self.assertEqual(retry.image_prompt.count("V28 REPAIR:"), 1)
        self.assertIn("display-free workspace", retry.image_prompt)

    def test_direction_extraction_is_exact(self) -> None:
        self.assertEqual(
            _extract_direction(FAILED_CANARY_PROMPT),
            "A researcher working on a computer with a new framework highlighted",
        )

    def test_page_normalization_is_shared_by_generation_and_review(self) -> None:
        normalized = _normalize_visual_intent("A screen showing the framework's open-source page")
        self.assertIn("unbranded workstation", normalized)
        self.assertIn("no readable interface content", normalized)
        self.assertNotIn("page", normalized.casefold())

    def test_scene_setups_are_deterministic_and_diverse(self) -> None:
        names = [_shot_setup(index)[0] for index in range(20)]
        self.assertEqual(names[:10], names[10:])
        self.assertGreaterEqual(len(set(names)), 10)

    def test_quality_renderer_uses_full_sdxl_cfg_not_lightning(self) -> None:
        class Plan:
            width = 704
            height = 1280

        renderer = SDXLQualityKeyframeGeneratorV29(Plan())
        self.assertEqual(renderer.steps, 30)
        self.assertEqual(renderer.guidance, 5.5)
        self.assertIn("stable-diffusion-xl-base-1.0", renderer.model_id)
        self.assertIn("cfg-5.5", renderer.model_id)
        source = Path("factory/production_visual_convergence_v29.py").read_text(encoding="utf-8")
        self.assertIn("negative_prompt=negative", source)
        self.assertIn("guidance_scale=self.guidance", source)
        self.assertNotIn("sdxl_lightning_8step_unet", source)

    def test_reviewer_contract_forbids_literal_page_demands(self) -> None:
        source = Path("factory/production_visual_semantic_review_v28.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Never demand a website, article page", source)
        self.assertIn("screens with abstract unreadable shapes", source)
        self.assertIn("setup_alignment", source)
        self.assertNotIn("displaying the open-source framework's page", source)

    def test_retry_architecture_preserves_approved_assets(self) -> None:
        source = Path("factory/production_visual_semantic_review_v28.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("approved_assets_preserved_across_retries", source)
        self.assertIn("regenerate_failed_only", source)
        self.assertIn("pending = set(failed)", source)
        self.assertNotIn('scene.image_prompt\n                            + ". REQUIRED CORRECTION', source)

    def test_runtime_installs_v29_after_v28_semantic_review(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_production_visual_semantic_review_v28()"),
            source.index("install_production_visual_convergence_v29()"),
        )
        self.assertLess(
            source.index("install_production_visual_convergence_v29()"),
            source.index("install_production_visual_prompt_cleanup_v29()"),
        )
        self.assertNotIn(
            "image_generator.generate_keyframes = visual_pipeline.generate_keyframes",
            source,
        )
        editorial = Path("factory/production_editorial_v28.py").read_text(encoding="utf-8")
        self.assertIn("visual_pipeline.generate_keyframes(expanded, keyframe_dir)", editorial)


if __name__ == "__main__":
    unittest.main()
