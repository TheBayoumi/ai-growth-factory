from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.production_visual_runtime_v28 import install_production_visual_runtime_v28
from factory.production_visual_semantic_review_v28 import (
    _base_director_prompt,
    _extract_direction,
    _normalize_visual_intent,
    _sanitize_negative_prompt,
    _sanitize_repair,
    _shot_setup,
    compile_semantic_generation_prompt_v28,
)


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

    def test_prompt_budget_is_hard_and_clip_safe(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_SCENE_14_PROMPT, word_budget=500)
        self.assertLessEqual(compiled.word_count, 58)
        self.assertEqual(compiled.word_budget, 58)
        self.assertEqual(compiled.word_count, len(compiled.compiled_prompt.rstrip(".").split()))

    def test_people_hands_screens_and_devices_are_not_negative_defects(self) -> None:
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

    def test_retry_markers_are_replaced_not_accumulated(self) -> None:
        base = _base_director_prompt(FAILED_SCENE_14_PROMPT)
        self.assertNotIn("V28 SCENE SETUP", base)
        self.assertNotIn("V28 REPAIR", base)
        self.assertNotIn("REQUIRED CORRECTION", base)
        self.assertIn("Supporting source-grounded visual direction", base)

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
        self.assertNotEqual(names[0], names[1])
        self.assertNotEqual(names[1], names[2])

    def test_eight_step_checkpoint_matches_step_count(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            import factory.production_visual_runtime_v28 as runtime

            previous = runtime._INSTALLED
            runtime._INSTALLED = False
            try:
                install_production_visual_runtime_v28()
                self.assertEqual(
                    os.environ["VISUAL_SDXL_LIGHTNING_CHECKPOINT"],
                    "sdxl_lightning_8step_unet.safetensors",
                )
                self.assertEqual(os.environ["VISUAL_SDXL_LIGHTNING_STEPS"], "8")
            finally:
                runtime._INSTALLED = previous

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
        self.assertNotIn("people, faces, portraits, bodies, hands, phones, screens", source)

    def test_runtime_installs_semantic_review_after_legacy_visual_quality(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_production_visual_quality()"),
            source.index("install_production_visual_semantic_review_v28()"),
        )
        self.assertLess(
            source.index("install_production_visual_runtime_v28()"),
            source.index("install_production_visual_semantic_review_v28()"),
        )


if __name__ == "__main__":
    unittest.main()
