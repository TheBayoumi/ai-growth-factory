from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from factory.production_visual_quality import (
    _caption_zone_is_exact_matte,
    _normalize_review_payload,
    strengthen_compiled_prompt,
)
from factory.visual_prompt_compiler import CompiledVisualPrompt


class ProductionVisualQualityTests(unittest.TestCase):
    def test_strengthened_prompt_blocks_v14_failure_modes(self) -> None:
        raw = CompiledVisualPrompt(
            director_prompt="Google AI dashboard poster with a woman holding a phone",
            compiled_prompt=(
                "Text-free cinematic editorial image. Google AI dashboard poster with a woman "
                "holding a phone. Subject high in frame. Dark empty lower third reserved for captions."
            ),
            negative_prompt="text, logo",
            word_count=18,
            word_budget=44,
        )
        result = strengthen_compiled_prompt(raw)
        lowered = result.compiled_prompt.casefold()
        self.assertIn("photorealistic conceptual scene", lowered)
        self.assertIn("no people", lowered)
        self.assertNotIn("google", lowered)
        self.assertNotIn("woman", lowered)
        self.assertNotIn("phone", lowered)
        self.assertNotIn("dashboard", lowered)
        self.assertNotIn("dashboard poster", lowered)
        self.assertIn("posters", lowered)
        self.assertIn("collage", result.negative_prompt.casefold())
        self.assertIn("pseudo-text", result.negative_prompt.casefold())

    def test_strengthened_prompt_preserves_concrete_mechanism_language(self) -> None:
        raw = CompiledVisualPrompt(
            director_prompt="Directional memory handoff",
            compiled_prompt=(
                "Text-free cinematic editorial image. A sequence of physical memory blocks moves "
                "through a directional handoff into a stable archive. Subject high in frame. "
                "Dark empty lower third reserved for captions."
            ),
            negative_prompt="text",
            word_count=29,
            word_budget=44,
        )
        result = strengthen_compiled_prompt(raw)
        self.assertIn("memory blocks", result.compiled_prompt.casefold())
        self.assertIn("directional handoff", result.compiled_prompt.casefold())
        self.assertTrue(result.compiler_version.endswith("+quality-v1"))

    def test_exact_lower_matte_is_proven_from_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keyframe.png"
            image = Image.new("RGB", (100, 100), (40, 70, 110))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 68, 99, 99), fill=(5, 7, 12))
            image.save(path)

            self.assertTrue(_caption_zone_is_exact_matte(path))

    def test_nonuniform_lower_matte_fails_deterministic_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keyframe.png"
            image = Image.new("RGB", (100, 100), (40, 70, 110))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 68, 99, 99), fill=(5, 7, 12))
            draw.point((50, 90), fill=(6, 7, 12))
            image.save(path)

            self.assertFalse(_caption_zone_is_exact_matte(path))

    def test_explicit_fields_override_inconsistent_retry_and_placeholder_feedback(self) -> None:
        review = _normalize_review_payload(
            {
                "decision": "retry",
                "claim_alignment": 0.84,
                "coherent_scene": True,
                "visible_text": "false",
                "prominent_person": False,
                "device_or_panel": False,
                "collage_layout": False,
                "reason": "specific visible defect or empty when approved",
                "repair_instruction": (
                    "standalone image-generation correction or empty when approved"
                ),
            },
            scene_index=3,
            attempt=1,
            caption_zone_clear=True,
            executable_prompt="One modular column rises into aligned glowing blocks.",
        )

        self.assertEqual(review.decision, "approve")
        self.assertTrue(review.caption_zone_clear)
        self.assertEqual(review.reason, "")
        self.assertEqual(review.repair_instruction, "")

    def test_low_alignment_remains_retry_with_executable_fallback(self) -> None:
        executable = "Connected modules expand across a clean architectural field."
        review = _normalize_review_payload(
            {
                "decision": "retry",
                "claim_alignment": 0.42,
                "coherent_scene": True,
                "visible_text": False,
                "prominent_person": False,
                "device_or_panel": False,
                "collage_layout": False,
                "reason": "specific visible defect or empty when approved",
                "repair_instruction": (
                    "standalone image-generation correction or empty when approved"
                ),
            },
            scene_index=4,
            attempt=2,
            caption_zone_clear=True,
            executable_prompt=executable,
        )

        self.assertEqual(review.decision, "retry")
        self.assertIn("claim alignment", review.reason)
        self.assertIn(executable, review.repair_instruction)
        self.assertNotIn("when approved", review.repair_instruction)

    def test_real_visual_defect_cannot_be_overridden_by_model_decision(self) -> None:
        review = _normalize_review_payload(
            {
                "decision": "approve",
                "claim_alignment": 0.91,
                "coherent_scene": True,
                "visible_text": True,
                "prominent_person": False,
                "device_or_panel": False,
                "collage_layout": False,
            },
            scene_index=0,
            attempt=1,
            caption_zone_clear=True,
            executable_prompt="A modular bridge joins separated luminous blocks.",
        )

        self.assertEqual(review.decision, "retry")
        self.assertTrue(review.visible_text)
        self.assertIn("unmarked surfaces", review.repair_instruction)

    def test_corrupted_caption_zone_cannot_be_approved(self) -> None:
        review = _normalize_review_payload(
            {
                "decision": "approve",
                "claim_alignment": 0.95,
                "coherent_scene": True,
                "visible_text": False,
                "prominent_person": False,
                "device_or_panel": False,
                "collage_layout": False,
            },
            scene_index=0,
            attempt=1,
            caption_zone_clear=False,
            executable_prompt="A modular bridge joins separated luminous blocks.",
        )

        self.assertEqual(review.decision, "retry")
        self.assertFalse(review.caption_zone_clear)
        self.assertIn("lower matte", review.reason)


if __name__ == "__main__":
    unittest.main()
