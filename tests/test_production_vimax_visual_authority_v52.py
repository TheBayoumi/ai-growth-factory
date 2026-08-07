from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_vimax_visual_authority_v52 import (
    _physicalized_direction,
    _physicalized_motion,
    _semantic_direction,
    compile_vimax_image_prompt_v52,
    validate_vimax_storyboard_v52,
)


class ViMaxVisualAuthorityV52Tests(unittest.TestCase):
    def test_text_card_is_physicalized_without_legacy_frame_bank(self) -> None:
        director = (
            "[VIMAX_SHOT_INDEX=3] "
            "Factual technology documentary shot synchronized to this exact spoken sentence: "
            "Baseten is now a supported inference provider on the Hugging Face Hub. "
            "Supporting source-grounded visual direction: "
            "A close-up shot of the text 'Baseten is now a supported inference provider on the Hugging Face Hub'. "
            "Shot treatment: close eye-level documentary framing. "
            "ViMax first frame: The text is displayed in white on a digital interface."
        )
        compiled = compile_vimax_image_prompt_v52(director)
        lowered = compiled.compiled_prompt.casefold()
        for phrase in ("baseten", "hugging face", "inference provider"):
            self.assertIn(phrase, lowered)
        for stale in ("robot", "thermal sensors", "factory sensor", "archival research"):
            self.assertNotIn(stale, lowered)
        self.assertEqual("visual-compiler-v52-vimax-authority", compiled.compiler_version)

    def test_non_text_vimax_direction_is_preserved(self) -> None:
        director = (
            "[VIMAX_SHOT_INDEX=4] "
            "A developer connects a compact inference appliance to a workstation while another engineer "
            "checks three model-serving nodes. Wide eye-level documentary view."
        )
        compiled = compile_vimax_image_prompt_v52(director)
        lowered = compiled.compiled_prompt.casefold()
        self.assertIn("developer", lowered)
        self.assertIn("inference appliance", lowered)
        self.assertIn("model-serving nodes", lowered)

    def test_storyboard_rejects_repeated_text_cards_and_static_motion(self) -> None:
        scenes = tuple(
            SimpleNamespace(
                image_prompt="A medium shot of a digital interface with text-based information about Baseten.",
                motion_prompt="Static camera. The interface remains in the same position with no significant changes.",
            )
            for _ in range(20)
        )
        with self.assertRaisesRegex(ValueError, "text/interface cards"):
            validate_vimax_storyboard_v52(SimpleNamespace(scenes=scenes))

    def test_semantic_direction_uses_quoted_claim_for_text_card(self) -> None:
        direction, textual = _semantic_direction(
            "A close-up shot of the text 'Provider routing with API keys and SDK integration'."
        )
        self.assertTrue(textual)
        self.assertEqual("Provider routing with API keys and SDK integration", direction)

    def test_text_card_physicalization_is_filmable_and_motionful(self) -> None:
        direction = _physicalized_direction(
            "Baseten can be selected as an inference provider for supported Hugging Face models.",
            7,
        )
        motion = _physicalized_motion(7)
        lowered = direction.casefold()
        self.assertIn("developer", lowered)
        self.assertIn("inference", lowered)
        self.assertIn("baseten", lowered)
        self.assertNotIn("digital interface", lowered)
        self.assertNotIn("shot of the text", lowered)
        self.assertNotRegex(motion.casefold(), r"static camera|no movement|unchanged")


if __name__ == "__main__":
    unittest.main()
