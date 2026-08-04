from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.production_visual_runtime_v28 import install_production_visual_runtime_v28
from factory.production_visual_semantic_review_v28 import (
    _extract_direction,
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


class ProductionVisualSemanticReviewV28Tests(unittest.TestCase):
    def test_generation_prompt_never_receives_narration_prose(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_CANARY_PROMPT)
        self.assertNotIn("Microsoft Research has launched", compiled.compiled_prompt)
        self.assertNotIn("training and evaluating AI agents", compiled.compiled_prompt)
        self.assertNotIn("Microsoft", compiled.compiled_prompt)
        self.assertIn("generic adult AI researcher", compiled.compiled_prompt)
        self.assertIn("real laboratory workspace", compiled.compiled_prompt)

    def test_people_and_unbranded_devices_are_not_negative_defects(self) -> None:
        compiled = compile_semantic_generation_prompt_v28(FAILED_CANARY_PROMPT)
        negative = compiled.negative_prompt.casefold()
        self.assertNotIn("people, faces, portraits, bodies", negative)
        self.assertNotIn("devices, screens", negative)
        self.assertIn("malformed anatomy", negative)
        self.assertIn("readable text", negative)
        self.assertIn("architecture-only scene", negative)

    def test_graph_direction_becomes_physical_evidence_not_infographic(self) -> None:
        prompt = FAILED_CANARY_PROMPT.replace(
            "A researcher working on a computer with a new framework highlighted",
            "A graph showing performance metrics for AI models",
        )
        compiled = compile_semantic_generation_prompt_v28(prompt)
        self.assertIn("physical AI evaluation bench", compiled.compiled_prompt)
        self.assertIn("ascending illuminated status columns", compiled.compiled_prompt)
        self.assertNotIn("graph showing", compiled.compiled_prompt.casefold())
        self.assertNotIn("infographic", compiled.compiled_prompt.casefold())

    def test_direction_extraction_is_exact(self) -> None:
        self.assertEqual(
            _extract_direction(FAILED_CANARY_PROMPT),
            "A researcher working on a computer with a new framework highlighted",
        )

    def test_repair_instruction_is_visual_only(self) -> None:
        prompt = FAILED_CANARY_PROMPT + (
            " REQUIRED CORRECTION: Use one coherent laboratory scene and remove pseudo-text."
        )
        compiled = compile_semantic_generation_prompt_v28(prompt)
        self.assertIn("Apply this correction", compiled.compiled_prompt)
        self.assertIn("remove pseudo-text", compiled.compiled_prompt.casefold())
        self.assertNotIn("Microsoft Research has launched", compiled.compiled_prompt)

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

    def test_reviewer_contract_explicitly_allows_researchers_and_devices(self) -> None:
        source = Path("factory/production_visual_semantic_review_v28.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Generic adult researchers", source)
        self.assertIn("A person or device is not a defect by itself", source)
        self.assertIn("generic corridor", source)
        self.assertIn("semantic alignment", source)
        self.assertNotIn("executable object-only brief used", source)

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
