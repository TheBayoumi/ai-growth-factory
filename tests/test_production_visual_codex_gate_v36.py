from __future__ import annotations

import inspect
import unittest
from dataclasses import dataclass

from factory import (
    image_generator,
    production_visual_codex_gate_v36 as codex_v36,
    production_visual_semantic_review_v28 as semantic_v28,
    visual_prompt_compiler,
    visual_storyboard_v30,
)
from factory.production_visual_codex_gate_v36 import (
    CodexVisualGateError,
    compile_codex_reviewed_prompt_v36,
    install_production_visual_codex_gate_v36,
    scene_contract_for_v36,
    scene_for_attempt_v36,
    validate_codex_visual_gate_v36,
)
from factory.visual_storyboard_v30 import storyboard_for


@dataclass(frozen=True)
class SceneFixture:
    image_prompt: str
    negative_prompt: str
    seed: int


def _director(scene_index: int, claim: str) -> str:
    return (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        f"{claim}. Supporting source-grounded visual direction: controlled physical evaluation. "
        f"Shot treatment: documentary view. V30 STORYBOARD: shot-{scene_index}"
    )


class ProductionVisualCodexGateV36Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._controlled_test_registry = visual_storyboard_v30._REGISTRY["controlled_test"]
        cls._visual_prompt_compiler = visual_prompt_compiler.compile_image_prompt
        cls._image_generator_compiler = image_generator.compile_image_prompt
        cls._scene_for_attempt = semantic_v28._scene_for_attempt
        cls._codex_installed = codex_v36._INSTALLED
        install_production_visual_codex_gate_v36()

    @classmethod
    def tearDownClass(cls) -> None:
        visual_storyboard_v30._REGISTRY["controlled_test"] = cls._controlled_test_registry
        visual_prompt_compiler.compile_image_prompt = cls._visual_prompt_compiler
        image_generator.compile_image_prompt = cls._image_generator_compiler
        semantic_v28._scene_for_attempt = cls._scene_for_attempt
        codex_v36._INSTALLED = cls._codex_installed

    def test_scene_16_preserves_exact_machine_contract_without_negative_conflict(self) -> None:
        compiled = compile_codex_reviewed_prompt_v36(
            _director(16, "Before adoption read the source and test the claim on a controlled task")
        )
        positive = compiled.compiled_prompt.casefold()
        negative = compiled.negative_prompt.casefold()
        for required in (
            "compact robotic gripper",
            "four blank calibration cubes",
            "short aluminum rail",
            "overhead optical sensor",
        ):
            self.assertIn(required, positive)
            self.assertNotIn(required, negative)
        for forbidden in ("dslr", "tripod", "product photography", "split view"):
            self.assertIn(forbidden, negative)
        self.assertLessEqual(compiled.word_count, compiled.word_budget)

    def test_scene_18_forbids_laptop_lamp_and_diagram_substitutions(self) -> None:
        compiled = compile_codex_reviewed_prompt_v36(
            _director(
                18,
                "Track latency failure rate human corrections and repeatability on a controlled task",
            )
        )
        positive = compiled.compiled_prompt.casefold()
        negative = compiled.negative_prompt.casefold()
        for required in (
            "adult evaluator",
            "two identical unlabelled benchtop motor-test fixtures",
            "physical start button",
            "embedded green and amber indicator leds",
        ):
            self.assertIn(required, positive)
        for forbidden in ("laptop", "desk lamp", "product display", "catalog illustration"):
            self.assertIn(forbidden, negative)
            self.assertNotIn(forbidden, positive)

    def test_codex_gate_rejects_required_phrase_loss_before_gpu_inference(self) -> None:
        director = _director(16, "Before adoption test the claim on a controlled task")
        contract = scene_contract_for_v36(storyboard_for(director, 16))
        with self.assertRaisesRegex(CodexVisualGateError, "missing"):
            validate_codex_visual_gate_v36(
                contract,
                "Photorealistic product photograph of one camera on a pedestal.",
                "tripod, laptop",
            )

    def test_retry_restarts_from_base_and_converts_reviewer_failure_into_negatives(self) -> None:
        scene = SceneFixture(
            image_prompt=_director(16, "Before adoption test the claim on a controlled task"),
            negative_prompt="legacy",
            seed=23,
        )
        retry = scene_for_attempt_v36(
            scene,
            scene_index=16,
            attempt=2,
            repair="The image contains a photography camera and tripod instead of the robotic fixture",
        )
        self.assertEqual(retry.image_prompt.count("V30 STORYBOARD:"), 1)
        self.assertEqual(retry.image_prompt.count("V31 REPAIR:"), 1)
        self.assertNotEqual(retry.seed, scene.seed)
        self.assertIn("photography camera", retry.negative_prompt.casefold())
        self.assertIn("tripod", retry.negative_prompt.casefold())

        compiled = compile_codex_reviewed_prompt_v36(retry.image_prompt)
        self.assertIn("compact robotic gripper", compiled.compiled_prompt.casefold())
        self.assertIn("four blank calibration cubes", compiled.compiled_prompt.casefold())
        self.assertIn("photography camera", compiled.negative_prompt.casefold())

    def test_v35_installer_hands_final_authority_to_v36(self) -> None:
        from factory import production_visual_retry_grounding_v35

        source = inspect.getsource(
            production_visual_retry_grounding_v35.install_production_visual_retry_grounding_v35
        )
        self.assertIn("install_production_visual_codex_gate_v36", source)


if __name__ == "__main__":
    unittest.main()
