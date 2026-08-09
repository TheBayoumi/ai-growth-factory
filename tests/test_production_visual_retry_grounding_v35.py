from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from factory.production_visual_retry_grounding_v35 import (
    grounded_retry_instruction_v35,
    scene_for_attempt_v35,
)
from factory.production_visual_subject_authority_v31 import compile_subject_first_prompt_v31


@dataclass(frozen=True)
class SceneFixture:
    image_prompt: str
    negative_prompt: str
    seed: int


class VisualRetryGroundingV35Tests(unittest.TestCase):
    def test_retry_instruction_preserves_subject_action_and_reviewer_feedback(self) -> None:
        frame = SimpleNamespace(
            subject="one adult researcher beside an unbranded workstation",
            action="the researcher connects a compact module to laboratory hardware",
        )
        instruction = grounded_retry_instruction_v35(
            frame=frame,
            reviewer_reason="The image lacks the required elements and environment",
        )
        lowered = instruction.casefold()
        self.assertIn("adult researcher", lowered)
        self.assertIn("connects a compact module", lowered)
        self.assertIn("reviewer correction", lowered)
        self.assertIn("named machine", lowered)

    def test_retry_scene_restarts_from_base_and_compiles_required_details(self) -> None:
        scene = SceneFixture(
            image_prompt=(
                "Supporting source-grounded visual direction: A researcher validates modular software. "
                "Shot treatment: human-scale documentary view."
            ),
            negative_prompt="legacy",
            seed=7,
        )
        retry = scene_for_attempt_v35(
            scene,
            scene_index=19,
            attempt=2,
            repair="The image does not match the storyboard and lacks required elements",
        )
        self.assertEqual(retry.image_prompt.count("V30 STORYBOARD:"), 1)
        self.assertEqual(retry.image_prompt.count("V31 REPAIR:"), 1)
        self.assertNotEqual(retry.seed, scene.seed)

        compiled = compile_subject_first_prompt_v31(retry.image_prompt)
        self.assertIn("REQUIRED SUBJECT AND ACTION", compiled.compiled_prompt)
        self.assertLessEqual(compiled.word_count, compiled.word_budget)


if __name__ == "__main__":
    unittest.main()
