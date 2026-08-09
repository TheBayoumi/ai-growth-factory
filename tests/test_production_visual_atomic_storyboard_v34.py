from __future__ import annotations

import inspect
import unittest

from factory.production_visual_atomic_storyboard_v34 import (
    compact_negative_atomic_v34,
    install_production_visual_atomic_storyboard_v34,
    physical_repair_atomic_v34,
    validate_atomic_controlled_test_registry_v34,
)
from factory.production_visual_subject_authority_v31 import (
    _words,
    compile_subject_first_prompt_v31,
)
from factory.visual_storyboard_v30 import storyboard_for


_SCENE_19_DIRECTOR = (
    "Factual technology documentary shot synchronized to this exact spoken sentence: "
    "Before adoption, read the linked source and test the claim on a controlled task. "
    "Supporting source-grounded visual direction: a researcher working on a computer. "
    "Shot treatment: human-scale consequence. V30 STORYBOARD: shot-19"
)


class ProductionVisualAtomicStoryboardV34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_production_visual_atomic_storyboard_v34()

    def test_exact_failed_scene_becomes_one_atomic_machine_action(self) -> None:
        frame = storyboard_for(_SCENE_19_DIRECTOR, 19)
        self.assertEqual(frame.identity, "controlled_test-1")
        combined = " ".join((frame.environment, frame.subject, frame.action)).casefold()
        for required in ("one compact robotic gripper", "one small overhead camera sensor", "one block"):
            self.assertIn(required, combined)
        for forbidden in ("two adult researchers", "microscope", "computer", "screen", "display"):
            self.assertNotIn(forbidden, combined)

    def test_scene_19_compiled_prompt_is_clip_safe_and_text_resistant(self) -> None:
        compiled = compile_subject_first_prompt_v31(_SCENE_19_DIRECTOR)
        lowered = compiled.compiled_prompt.casefold()
        self.assertLessEqual(len(_words(compiled.compiled_prompt)), 52)
        self.assertIn("robotic gripper", lowered)
        self.assertIn("blank blocks", lowered)
        self.assertIn("camera sensor", lowered)
        self.assertNotIn("researchers", lowered)
        self.assertNotIn("microscope", lowered)
        self.assertIn("pseudo-text", compiled.negative_prompt)
        self.assertIn("printed label", compiled.negative_prompt)
        self.assertNotIn("absent people", compiled.negative_prompt)

    def test_missing_element_retry_does_not_invent_people(self) -> None:
        repair = physical_repair_atomic_v34(
            "The image lacks the required robotic gripper, camera sensor, and block row"
        ).casefold()
        self.assertIn("named machine", repair)
        self.assertIn("single physical action", repair)
        self.assertNotIn("adult", repair)
        self.assertNotIn("people", repair)

    def test_controlled_test_registry_has_atomic_complexity_guard(self) -> None:
        validate_atomic_controlled_test_registry_v34()

    def test_compact_negative_stays_inside_conservative_word_budget(self) -> None:
        negative = compact_negative_atomic_v34()
        self.assertLessEqual(len(_words(negative)), 36)
        for required in ("pseudo-text", "gibberish", "printed label", "engraved markings"):
            self.assertIn(required, negative)

    def test_runtime_installs_v34_after_v31(self) -> None:
        from factory import production_runtime

        source = inspect.getsource(production_runtime.install_production_runtime)
        self.assertLess(
            source.index("install_production_visual_subject_authority_v31()"),
            source.index("install_production_visual_atomic_storyboard_v34()"),
        )


if __name__ == "__main__":
    unittest.main()
