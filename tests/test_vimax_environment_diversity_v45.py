from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory import production_vimax_planning_v45 as vimax_v45
from factory import production_visual_convergence_v41 as convergence_v41


class ViMaxEnvironmentDiversityV45Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_business_frames = tuple(
            convergence_v41._FRAME_BANKS["business_adoption"]
        )
        self.original_named_frames = convergence_v41._DIVERSE_BUSINESS_FRAMES

    def tearDown(self) -> None:
        convergence_v41._DIVERSE_BUSINESS_FRAMES = self.original_named_frames
        convergence_v41._FRAME_BANKS["business_adoption"] = (
            self.original_business_frames
        )

    @staticmethod
    def _business_scene(index: int) -> SimpleNamespace:
        prompt = (
            "Factual technology documentary shot synchronized to this exact spoken sentence: "
            "Companies deploy the model across enterprise use cases and business workloads. "
            "Supporting source-grounded visual direction: Show one concrete enterprise "
            "deployment workflow. Shot treatment: human-scale consequence. "
            f"V30 STORYBOARD: shot-{index}; validation"
        )
        return SimpleNamespace(scene_index=index, image_prompt=prompt)

    def test_twenty_business_shots_use_five_balanced_families(self) -> None:
        vimax_v45._install_business_environment_diversity_v45()

        families = convergence_v41.validate_editorial_contract_diversity_v41(
            self._business_scene(index) for index in range(20)
        )

        self.assertEqual(len(families), 5)
        self.assertEqual(set(families.values()), {4})
        self.assertIn("field-mobile", families)

    def test_environment_extension_is_idempotent(self) -> None:
        vimax_v45._install_business_environment_diversity_v45()
        vimax_v45._install_business_environment_diversity_v45()

        frames = convergence_v41._FRAME_BANKS["business_adoption"]
        field_frames = [
            frame
            for frame in frames
            if "portable field-service" in frame.environment.casefold()
        ]
        self.assertEqual(len(frames), len(self.original_business_frames) + 1)
        self.assertEqual(len(field_frames), 1)


if __name__ == "__main__":
    unittest.main()
