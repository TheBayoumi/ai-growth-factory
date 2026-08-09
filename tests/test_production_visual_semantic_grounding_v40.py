from __future__ import annotations

import unittest
from dataclasses import dataclass

from factory.production_visual_semantic_grounding_v40 import (
    classify_grounded_scene_v40,
    compile_grounded_prompt_v40,
    grounded_contract_for_v40,
    scene_for_attempt_v40,
    storyboard_frame_for_v40,
)


_TREATMENTS = (
    "tight concrete detail with a clear foreground action",
    "wide contextual view showing the surrounding workflow",
    "cause-to-result process view with visible directional change",
    "human-scale consequence using generic unbranded people or tools when relevant",
    "clean comparison view with two visibly different arrangements",
)


def _director(claim: str, direction: str, index: int, *, treatment: str | None = None) -> str:
    return (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        f"{claim}. Supporting source-grounded visual direction: {direction}. "
        f"Shot treatment: {treatment or _TREATMENTS[index % len(_TREATMENTS)]}. "
        f"V30 STORYBOARD: shot-{index}; legacy-marker"
    )


@dataclass(frozen=True)
class _Scene:
    image_prompt: str
    negative_prompt: str
    seed: int


class ProductionVisualSemanticGroundingV40Tests(unittest.TestCase):
    def test_local_llm_claim_cannot_fall_back_to_partnership_hub(self) -> None:
        director = _director(
            "Hugging Face has introduced a new approach to deploying AI models locally",
            "A screen showing a model deployment interface with local infrastructure highlighted",
            0,
        )
        contract = grounded_contract_for_v40(director)
        compiled = compile_grounded_prompt_v40(director)

        self.assertEqual(contract.category, "local_inference")
        self.assertIn("local AI inference", compiled.compiled_prompt)
        self.assertIn("private", compiled.compiled_prompt.casefold())
        self.assertNotIn("partnership hub", compiled.compiled_prompt.casefold())
        self.assertNotIn("robotics bay", compiled.compiled_prompt.casefold())
        self.assertNotIn("automation kit", compiled.compiled_prompt.casefold())
        self.assertLessEqual(compiled.word_count, compiled.word_budget)

    def test_own_infrastructure_claim_maps_to_local_inference(self) -> None:
        director = _director(
            "This strategy allows users to run large language models on their own infrastructure",
            "A user adjusting model settings in a custom interface",
            2,
        )
        frame = storyboard_frame_for_v40(director)
        self.assertEqual(frame.category, "local_inference")
        self.assertTrue(
            any(term in frame.environment.casefold() for term in ("private", "local", "edge"))
        )

    def test_release_initiative_is_model_release_not_partnership(self) -> None:
        director = _director(
            "The release is part of a larger initiative to enhance AI capabilities",
            "A newly released model being prepared for deployment",
            19,
        )
        self.assertEqual(grounded_contract_for_v40(director).category, "model_release")
        self.assertEqual(
            classify_grounded_scene_v40(
                "The release is part of a larger initiative to enhance AI capabilities"
            ),
            "model_release",
        )

    def test_unknown_software_claim_has_safe_source_grounded_fallback(self) -> None:
        self.assertEqual(
            classify_grounded_scene_v40("A new software workflow changes model processing"),
            "source_grounded_ai",
        )

    def test_twenty_one_local_shots_compile_to_twenty_one_distinct_briefs(self) -> None:
        prompts = []
        for index in range(21):
            director = _director(
                "Organizations can deploy AI models locally on their own infrastructure",
                "Private local AI deployment hardware",
                index,
            )
            compiled = compile_grounded_prompt_v40(director)
            prompts.append(compiled.compiled_prompt)
            self.assertLessEqual(compiled.word_count, 58)
        self.assertEqual(len(set(prompts)), 21)

    def test_retry_preserves_topic_and_reviewer_feedback(self) -> None:
        scene = _Scene(
            image_prompt=_director(
                "By deploying models locally organizations can maintain control over their data and processing",
                "A local inference server connected to private storage",
                6,
            ),
            negative_prompt="",
            seed=31,
        )
        retried = scene_for_attempt_v40(
            scene,
            scene_index=6,
            attempt=2,
            repair="The image shows a robotics workshop instead of a private local inference server",
        )
        contract = grounded_contract_for_v40(retried.image_prompt, 6)
        compiled = compile_grounded_prompt_v40(retried.image_prompt)

        self.assertEqual(contract.category, "local_inference")
        self.assertIn("robotics workshop", retried.image_prompt.casefold())
        self.assertIn("robotics workshop", retried.negative_prompt.casefold())
        self.assertIn("local AI inference", compiled.compiled_prompt)
        self.assertNotEqual(retried.seed, scene.seed)

    def test_required_topic_is_never_negated(self) -> None:
        director = _director(
            "The new method improves efficiency and reduces reliance on cloud services",
            "A private local server processing an AI workload",
            4,
        )
        contract = grounded_contract_for_v40(director)
        compiled = compile_grounded_prompt_v40(director)
        for phrase in contract.required_phrases:
            self.assertNotIn(phrase.casefold(), compiled.negative_prompt.casefold())


if __name__ == "__main__":
    unittest.main()
