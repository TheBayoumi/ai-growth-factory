from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
import factory.production_vimax_human_editorial_v66 as v66


class ProductionViMaxHumanEditorialV66IntegrationTests(unittest.TestCase):
    @staticmethod
    def _source() -> SourceItem:
        return SourceItem(
            publisher="NVIDIA",
            author="Rev Lebaredian",
            title="Firebird Launches CIS Region's Largest AI Factory in Armenia",
            url="https://example.com/firebird",
            summary="Firebird launched a large AI factory in Armenia with dedicated compute, cooling, networking, power and local engineering capacity.",
            published_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _bad_package() -> VideoPackage:
        return VideoPackage(
            topic="Firebird AI factory",
            narration=(
                "Firebird launched a new AI factory in Armenia. NVIDIA provide separate primary-source context for this topic. "
                "Each report is evaluated independently and supports only its attributed claim. "
                + "Concrete infrastructure evidence follows. " * 22
            ),
            title="Firebird Launches AI Factory",
            description="Source-backed report.",
            tags=["AI", "Armenia", "compute", "infrastructure", "engineering", "data center", "research", "deployment"],
            thumbnail_text="AI FACTORY",
            top_comment="What matters most?",
            scenes=[
                Scene(
                    heading="AI Factory Launch",
                    body="NVIDIA launches CIS region's largest AI factory in Armenia.",
                    visual="Infrastructure documentary scene.",
                    source_index=0,
                )
                for _ in range(6)
            ],
            source_urls=["https://example.com/firebird"],
            source_publishers=["NVIDIA"],
        )

    def test_final_source_attributed_canary_boundary_rejects_internal_copy(self) -> None:
        from factory import canary, local_llm, source_attributed_llm

        source = self._source()
        bad = self._bad_package()
        with patch.object(source_attributed_llm, "generate_package", return_value=bad), patch.object(
            canary,
            "generate_package",
            source_attributed_llm.generate_package,
        ), patch.object(local_llm, "_chat", side_effect=AssertionError("chat not expected")), patch.object(
            local_llm,
            "_package_from_raw",
            side_effect=AssertionError("package parser not expected"),
        ), patch.object(local_llm, "_repair_prompt", return_value="repair"):
            v66._install_consumer_copy_gate()
            with self.assertRaisesRegex(Exception, "Human editorial copy gate failed"):
                canary.generate_package(SimpleNamespace(), [source], SimpleNamespace())

    def test_package_prompt_detection_covers_initial_and_repair_prompts(self) -> None:
        self.assertTrue(v66._is_package_prompt("SOURCE ENTRIES:\nReturn one JSON object containing:"))
        self.assertTrue(v66._is_package_prompt("SOURCE ENTRIES:\nPREVIOUS JSON:"))
        self.assertFalse(v66._is_package_prompt("scene attribution only"))


if __name__ == "__main__":
    unittest.main()
