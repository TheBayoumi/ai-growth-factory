from __future__ import annotations

import unittest
from pathlib import Path

from factory.config import Settings
from factory.local_llm import LocalLLMError
from factory.models import Scene, VideoPackage
from factory.production_editorial_v28 import ProductionPreflightError
from factory.production_spoken_duration_capacity_v51 import (
    maximum_spoken_equivalents_for_profile_v51,
    projected_narration_segments_v51,
    spoken_equivalent_count_v51,
    validate_spoken_duration_capacity_v51,
)
from factory.video_profile import VideoProfile


_FAILED_HSP_NARRATION = (
    "HSP GRUPPE, a European tax advisory network, has integrated ChatGPT Enterprise to improve "
    "productivity and client service. The firm reports 98.6% higher productivity and 84% weekly "
    "usage. AI is reshaping professional work by embedding into tax advisory and client "
    "communication. The system processes 500,000+ conversations in six months, with 40,000+ "
    "annual hours of additional capacity. AI is used within strict data protection and "
    "confidentiality guidelines. HSP GRUPPE has reinvented professional work by embedding AI into "
    "tax advisory, legal research, and client communication. The firm emphasizes adoption, "
    "governance, and continuous learning with monthly forums. ChatGPT Enterprise is a powerful "
    "tool that enables tax professionals to process information faster, improve work quality, and "
    "create more capacity for advisory work and client service. Track latency, failure rate, human "
    "corrections, and repeatability so the decision follows measured behavior rather than a "
    "polished announcement."
)


class SpokenDurationCapacityV51Tests(unittest.TestCase):
    @staticmethod
    def _package(narration: str) -> VideoPackage:
        return VideoPackage(
            topic="HSP GRUPPE ChatGPT Enterprise",
            narration=narration,
            title="HSP GRUPPE Measures ChatGPT Enterprise Adoption",
            description="Source-backed evidence.",
            tags=["AI"],
            thumbnail_text="HSP AI ADOPTION",
            top_comment="What would you verify?",
            scenes=[
                Scene(
                    heading=f"Evidence {index}",
                    body=f"Measured workflow evidence {index}",
                    visual=f"Literal workflow {index}",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/hsp"],
            source_publishers=["HSP GRUPPE"],
        )

    def test_exact_failed_script_is_139_written_but_150_spoken_equivalents(self) -> None:
        package = self._package(_FAILED_HSP_NARRATION)
        self.assertEqual(len(package.narration.split()), 139)
        self.assertEqual(spoken_equivalent_count_v51(package.narration), 150.0)

    def test_exact_failed_script_is_rejected_before_tts(self) -> None:
        package = self._package(_FAILED_HSP_NARRATION)
        with self.assertRaisesRegex(
            LocalLLMError,
            "exceeds spoken-duration budget: 150.0 spoken-word equivalents",
        ):
            validate_spoken_duration_capacity_v51(package)

    def test_plain_132_word_script_stays_inside_spoken_budget(self) -> None:
        narration = " ".join(f"word{index}" for index in range(132))
        package = self._package(narration)
        self.assertEqual(validate_spoken_duration_capacity_v51(package), 132.0)

    def test_spoken_aware_static_preflight_rejects_failed_script_before_tts(self) -> None:
        settings = Settings.from_env()
        profile = VideoProfile()
        package = self._package(_FAILED_HSP_NARRATION)

        with self.assertRaisesRegex(
            ProductionPreflightError,
            "Projected spoken-equivalent narration duration",
        ):
            projected_narration_segments_v51(settings, package, profile)

    def test_plain_132_word_projection_fits_frozen_window(self) -> None:
        narration = " ".join(f"word{index}" for index in range(132))
        package = self._package(narration)
        segments, duration = projected_narration_segments_v51(
            Settings.from_env(),
            package,
            VideoProfile(),
        )
        self.assertTrue(segments)
        self.assertGreaterEqual(duration, 55.0)
        self.assertLessEqual(duration, 62.0)

    def test_conservative_140_budget_has_margin_below_mathematical_ceiling(self) -> None:
        mathematical = maximum_spoken_equivalents_for_profile_v51(
            VideoProfile(),
            segment_count=8,
        )
        self.assertGreaterEqual(mathematical, 141)
        self.assertLess(140, mathematical)

    def test_runtime_restores_v46_before_v51(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        v46 = source.index("install_production_package_capacity_v46()")
        v51 = source.index("install_production_spoken_duration_capacity_v51()")
        self.assertLess(v46, v51)


if __name__ == "__main__":
    unittest.main()
