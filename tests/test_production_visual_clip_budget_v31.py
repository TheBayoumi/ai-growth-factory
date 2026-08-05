from __future__ import annotations

import unittest

from factory.production_visual_clip_budget_v31 import compact_negative_clip_safe_v31
from factory.production_visual_subject_authority_v31 import (
    _compact_negative,
    _words,
    compile_subject_first_prompt_v31,
    validate_clip_windows,
)


LEGACY_NEGATIVE = (
    "readable text, pseudo-text, gibberish, logo, watermark, screen, monitor, poster, chart, "
    "infographic, collage, split frame, empty room, vacant server aisle, architecture-only scene, "
    "unoccupied workspace, absent people, tiny distant people, humanoid robot, duplicate people, "
    "malformed anatomy, extra limbs, distorted hands, warped equipment, blurry face, generic corridor, "
    "generic blocks, generic orb"
)


class _ConservativeTokenizer:
    model_max_length = 77

    def __call__(self, value: str, **_kwargs: object) -> dict[str, list[int]]:
        words = len(value.split())
        count = words + 7 if value.startswith("Photorealistic") else words * 2 + 2
        return {"input_ids": list(range(count))}


class _ConservativePipeline:
    tokenizer = _ConservativeTokenizer()
    tokenizer_2 = _ConservativeTokenizer()


class VisualClipBudgetV31Tests(unittest.TestCase):
    def test_package_boundary_installs_the_text_resistant_compact_contract(self) -> None:
        self.assertEqual(_compact_negative(), compact_negative_clip_safe_v31())
        self.assertLessEqual(len(_words(_compact_negative())), 36)
        self.assertNotIn("architecture-only", _compact_negative())
        self.assertIn("pseudo-text", _compact_negative())
        self.assertIn("gibberish", _compact_negative())
        self.assertIn("printed label", _compact_negative())
        self.assertNotIn("absent people", _compact_negative())

    def test_compact_negative_passes_a_conservative_two_token_per_word_budget(self) -> None:
        director = (
            "Factual technology documentary shot synchronized to this exact spoken sentence: "
            "Shared infrastructure helps smaller models achieve strong performance. "
            "Supporting source-grounded visual direction: a legacy screen request. "
            "Shot treatment: wide contextual view showing the surrounding workflow. "
            "V30 STORYBOARD: shot-5"
        )
        compiled = compile_subject_first_prompt_v31(director)
        validate_clip_windows(
            _ConservativePipeline(),
            compiled.compiled_prompt,
            compiled.negative_prompt,
        )

    def test_exact_legacy_negative_shape_would_fail_before_gpu_inference(self) -> None:
        with self.assertRaisesRegex(Exception, "negative=.*?/77"):
            validate_clip_windows(
                _ConservativePipeline(),
                "Photorealistic people perform one clear physical action.",
                LEGACY_NEGATIVE,
            )

    def test_high_value_defects_remain_in_the_compact_contract(self) -> None:
        negative = compact_negative_clip_safe_v31()
        for required in (
            "readable text",
            "pseudo-text",
            "gibberish",
            "printed label",
            "vacant scene",
            "malformed anatomy",
            "bad hands",
            "warped equipment",
            "corridor",
        ):
            self.assertIn(required, negative)


if __name__ == "__main__":
    unittest.main()
