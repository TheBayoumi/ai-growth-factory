from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from factory import production_visual_subject_authority_v31 as subject_v31
from factory.production_visual_convergence_v41 import grounded_contract_for_v41
from factory.production_visual_negative_budget_v44 import (
    _MAX_NEGATIVE_WORDS,
    bounded_negative_prompt_v44,
    compile_negative_budget_v44,
)


_SCENE_4 = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "This makes it ideal for environments with strict data privacy requirements. "
    "Supporting source-grounded visual direction: A secure data center scene. "
    "Shot treatment: cause-to-result process view with visible directional change. "
    "V30 STORYBOARD: shot-4; validation"
)


class _ConservativeTokenizer:
    model_max_length = 77

    def __call__(self, value: str, **_kwargs: object) -> dict[str, list[int]]:
        # Charge two tokens per simple word plus both special tokens. This is deliberately stricter
        # than the actual scene-4 ratio that produced 116 tokens from the old unbounded prompt.
        words = re.findall(r"[A-Za-z0-9]+", value)
        return {"input_ids": list(range(2 + 2 * len(words)))}


class ProductionVisualNegativeBudgetV44Tests(unittest.TestCase):
    def test_exact_scene_four_negative_contract_is_bounded(self) -> None:
        compiled = compile_negative_budget_v44(_SCENE_4)
        words = compiled.negative_prompt.split()

        self.assertLessEqual(len(words), _MAX_NEGATIVE_WORDS)
        self.assertEqual(
            compiled.compiler_version,
            "visual-compiler-v44-priority-negative-budget",
        )
        for phrase in (
            "readable text lettering numbers",
            "logo watermark",
            "collage split frame",
        ):
            self.assertIn(phrase, compiled.negative_prompt.casefold())

    def test_bounded_negative_fits_conservative_clip_estimate(self) -> None:
        negative = compile_negative_budget_v44(_SCENE_4).negative_prompt
        count = subject_v31._token_count(_ConservativeTokenizer(), negative)
        self.assertLessEqual(count, 77)

    def test_reviewer_observed_substitutions_have_highest_priority(self) -> None:
        contract = grounded_contract_for_v41(_SCENE_4)
        negative = bounded_negative_prompt_v44(
            contract,
            reviewer_feedback=(
                "The result is an outdoor mountain landscape with a spacesuit, weapon, "
                "bicycle, and multi-panel collage"
            ),
        ).casefold()
        for phrase in (
            "outdoor mountain landscape",
            "spacesuit astronaut armor",
            "weapon gun rifle",
            "bicycle motorcycle",
            "multi-panel contact sheet",
        ):
            self.assertIn(phrase, negative)
        self.assertLessEqual(len(negative.split()), _MAX_NEGATIVE_WORDS)

    def test_required_positive_subject_is_not_negated(self) -> None:
        contract = grounded_contract_for_v41(_SCENE_4)
        negative = bounded_negative_prompt_v44(contract)
        for required in contract.required_phrases:
            self.assertNotIn(required.casefold(), negative.casefold())

    def test_runtime_installs_v44_after_v41(self) -> None:
        source = Path("factory/production_visual_retry_grounding_v35.py").read_text(
            encoding="utf-8"
        )
        v41 = source.index("install_production_visual_convergence_v41()")
        v44 = source.index("install_production_visual_negative_budget_v44()")
        self.assertGreater(v44, v41)


if __name__ == "__main__":
    unittest.main()
