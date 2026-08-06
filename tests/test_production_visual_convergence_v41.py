from __future__ import annotations

import unittest
from dataclasses import dataclass

from factory.production_visual_convergence_v41 import (
    classify_scene_v41,
    compile_convergent_prompt_v41,
    grounded_contract_for_v41,
    scene_for_attempt_v41,
)


@dataclass(frozen=True)
class _Scene:
    image_prompt: str
    negative_prompt: str
    seed: int


def _director(claim: str, direction: str, index: int) -> str:
    return (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        f"{claim}. Supporting source-grounded visual direction: {direction}. "
        "Shot treatment: human-scale consequence using generic unbranded people or tools when relevant. "
        f"V30 STORYBOARD: shot-{index}; legacy-marker"
    )


class ProductionVisualConvergenceV41Tests(unittest.TestCase):
    def test_maximum_editorial_plan_has_unique_executable_contracts(self) -> None:
        identities = [
            grounded_contract_for_v41(
                _director(
                    "A source-grounded AI capability changes a concrete local workflow",
                    "A literal documentary view of the measured workflow",
                    index,
                )
            ).identity
            for index in range(30)
        ]
        self.assertEqual(len(identities), len(set(identities)))

    def test_enterprise_scene_uses_literal_indoor_network_action(self) -> None:
        director = _director(
            "Built on LiquidAI's technology, the model supports a wide range of applications, from custom solutions to enterprise use cases",
            "A technical diagram of the model",
            8,
        )
        contract = grounded_contract_for_v41(director)
        compiled = compile_convergent_prompt_v41(director)
        prompt = compiled.compiled_prompt.casefold()

        self.assertEqual(contract.category, "business_adoption")
        self.assertIn("office it technician", prompt)
        self.assertIn("blue ethernet cable", prompt)
        self.assertIn("short server rack", prompt)
        self.assertIn("one uninterrupted photograph", prompt)
        self.assertNotIn("deployment engineer", prompt)
        self.assertNotIn("rugged ai node", prompt)
        self.assertLessEqual(compiled.word_count, compiled.word_budget)

    def test_enterprise_terms_classify_before_generic_fallback(self) -> None:
        self.assertEqual(
            classify_scene_v41(
                "The model supports custom solutions and enterprise use cases"
            ),
            "business_adoption",
        )

    def test_exact_scene_eight_reviewer_defects_become_negatives(self) -> None:
        scene = _Scene(
            image_prompt=_director(
                "Built on LiquidAI's technology, the model supports a wide range of applications, from custom solutions to enterprise use cases",
                "A technical diagram of the model",
                8,
            ),
            negative_prompt="",
            seed=11,
        )
        retried = scene_for_attempt_v41(
            scene,
            scene_index=8,
            attempt=2,
            repair=(
                "The image shows a mountainous outdoor landscape with a person in a spacesuit "
                "holding a weapon beside a bicycle in a multi-panel collage"
            ),
        )
        negative = retried.negative_prompt.casefold()
        for phrase in (
            "outdoor mountain landscape",
            "spacesuit astronaut armor",
            "weapon gun rifle",
            "bicycle motorcycle",
            "multi-panel contact sheet",
        ):
            self.assertIn(phrase, negative)
        self.assertNotEqual(retried.seed, scene.seed)

    def test_controlled_scene_is_literal_tabletop_cycle(self) -> None:
        director = _director(
            "Before adoption, read the linked source and test the claim on a controlled task",
            "A user running an AI task locally",
            17,
        )
        contract = grounded_contract_for_v41(director)
        compiled = compile_convergent_prompt_v41(director)
        prompt = compiled.compiled_prompt.casefold()

        self.assertEqual(contract.category, "controlled_test")
        for phrase in (
            "tabletop quality-control test",
            "metal cylinder",
            "green button",
            "left tray",
            "one uninterrupted photograph",
        ):
            self.assertIn(phrase, prompt)
        self.assertNotIn("measurable physical outcomes", prompt)
        self.assertNotIn("motor rig", prompt)

    def test_controlled_scene_rejects_observed_factory_substitutions(self) -> None:
        scene = _Scene(
            image_prompt=_director(
                "Before adoption, read the linked source and test the claim on a controlled task",
                "A user running an AI task locally",
                17,
            ),
            negative_prompt="",
            seed=23,
        )
        retried = scene_for_attempt_v41(
            scene,
            scene_index=17,
            attempt=3,
            repair=(
                "The image shows multiple workers operating large industrial machinery "
                "on a factory floor in a black and white photograph"
            ),
        )
        negative = retried.negative_prompt.casefold()
        self.assertIn("factory floor heavy machinery", negative)
        self.assertIn("multiple workers", negative)
        self.assertIn("black and white vintage photograph", negative)


if __name__ == "__main__":
    unittest.main()
