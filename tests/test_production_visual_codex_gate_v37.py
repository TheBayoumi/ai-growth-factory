from __future__ import annotations

import inspect
import unittest
from dataclasses import dataclass

from factory import production_visual_codex_gate_v36 as codex_v36
from factory.production_visual_codex_gate_v37 import (
    enrich_visual_review_v37,
    validate_codex_visual_gate_v37,
)


@dataclass(frozen=True)
class ReviewFixture:
    decision: str
    reason: str
    repair_instruction: str


class ProductionVisualCodexGateV37Tests(unittest.TestCase):
    def test_punctuation_normalization_preserves_exact_partnership_subject(self) -> None:
        subject = (
            "three adult technical staff beside rugged compute cases, sensor kits, "
            "and a mobile robotics cart"
        )
        action = (
            "the staff unload one sensor case and connect it to the mobile robotics cart"
        )
        contract = codex_v36.SceneContractV36(
            identity="partnership_hub-2",
            environment="a regional applied engineering hub",
            subject=subject,
            action=action,
            camera="wide documentary photograph",
            required_phrases=(subject, action),
            forbidden_substitutions=(),
        )
        compiled = (
            "Photorealistic vertical technology documentary photograph three adult technical "
            "staff beside rugged compute cases sensor kits and a mobile robotics cart the staff "
            "unload one sensor case and connect it to the mobile robotics cart."
        )
        validate_codex_visual_gate_v37(contract, compiled, "logo, watermark")

    def test_token_normalized_gate_still_rejects_real_required_phrase_loss(self) -> None:
        contract = codex_v36.SceneContractV36(
            identity="partnership_hub-2",
            environment="a regional applied engineering hub",
            subject="three adult technical staff beside rugged compute cases",
            action="the staff connect one sensor case to a mobile robotics cart",
            camera="wide documentary photograph",
            required_phrases=("rugged compute cases", "mobile robotics cart"),
            forbidden_substitutions=(),
        )
        with self.assertRaisesRegex(codex_v36.CodexVisualGateError, "rugged compute cases"):
            validate_codex_visual_gate_v37(
                contract,
                "Photorealistic staff beside one mobile robotics cart.",
                "logo, watermark",
            )

    def test_reviewer_reason_is_preserved_for_failed_scene_regeneration(self) -> None:
        review = ReviewFixture(
            decision="retry",
            reason="The image contains a photography camera and tripod instead of the robotic fixture",
            repair_instruction="make the required physical action unmistakable",
        )
        enriched = enrich_visual_review_v37(review)
        self.assertIn("photography camera and tripod", enriched.repair_instruction)
        self.assertIn("physical action unmistakable", enriched.repair_instruction)

        generic_contract = codex_v36.SceneContractV36(
            identity="generic-0",
            environment="laboratory",
            subject="robotic fixture",
            action="the fixture moves one part",
            camera="medium photograph",
            required_phrases=("robotic fixture",),
            forbidden_substitutions=(),
        )
        negative = codex_v36.scene_negative_prompt_v36(
            generic_contract,
            reviewer_feedback=enriched.repair_instruction,
        ).casefold()
        self.assertIn("photography camera", negative)

    def test_v35_installer_hands_final_authority_to_v37(self) -> None:
        from factory import production_visual_retry_grounding_v35

        source = inspect.getsource(
            production_visual_retry_grounding_v35.install_production_visual_retry_grounding_v35
        )
        self.assertIn("install_production_visual_codex_gate_v36", source)
        self.assertIn("install_production_visual_codex_gate_v37", source)


if __name__ == "__main__":
    unittest.main()
