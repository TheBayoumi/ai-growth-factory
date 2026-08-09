from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
import factory.production_vimax_human_editorial_v66 as v66


_FIXED_NARRATION = (
    "Firebird launched a new AI factory in Armenia, adding computing capacity for local engineering and research teams. "
    "The facility combines compute with cooling, networking, and power at one operating site. "
    "Engineers can use that capacity for model training, inference, simulation, and controlled workloads. "
    "Local developers gain closer access to infrastructure that might otherwise require remote capacity or smaller shared systems. "
    "The deployment creates practical work commissioning racks, connecting fiber, validating cooling, and maintaining power delivery. "
    "Teams can test applications against dedicated hardware inside one managed facility. "
    "The project gives Armenia a larger base for AI engineering without implying infrastructure alone guarantees useful results. "
    "Teams still need suitable data, software, evaluation methods, and operating discipline. "
    "The useful question is whether the capacity produces reliable workloads and completed projects. "
    "Watch utilization, availability, and engineering outcomes before judging the facility by headline scale."
)


class ProductionViMaxHumanEditorialV66IntegrationTests(unittest.TestCase):
    @staticmethod
    def _source() -> SourceItem:
        return SourceItem(
            publisher="NVIDIA",
            author="Rev Lebaredian",
            title="Firebird Launches CIS Region's Largest AI Factory in Armenia",
            url="https://example.com/firebird",
            summary=(
                "Firebird launched a large AI factory in Armenia with dedicated compute infrastructure for local engineering and research teams. "
                "The facility combines accelerated computing with cooling, networking, power delivery, commissioning work, and managed operations. "
                "The source describes practical deployment activity, local access to computing capacity, infrastructure maintenance, and technical workloads including training and inference."
            ),
            published_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _bad_package() -> VideoPackage:
        return VideoPackage(
            topic="Firebird AI factory",
            narration=(
                "Firebird launched a new AI factory in Armenia. NVIDIA provide separate primary-source context for this topic. "
                "Each report is evaluated independently and supports only its attributed claim. "
                + "Concrete infrastructure evidence follows for local engineering teams. " * 18
            ),
            title="Firebird Launches AI Factory",
            description="Source-backed report.\nhttps://example.com/firebird",
            tags=["AI", "Armenia", "compute", "infrastructure", "engineering", "data center", "research", "deployment"],
            thumbnail_text="AI FACTORY",
            top_comment="What matters most?",
            scenes=[
                Scene(
                    heading=f"Original Beat {index}",
                    body=f"NVIDIA launches infrastructure claim number {index} for the Armenia facility.",
                    visual=f"Immutable visual direction {index}.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/firebird"],
            source_publishers=["NVIDIA"],
        )

    @staticmethod
    def _focused_rewrite() -> dict[str, object]:
        bodies = [
            "Firebird's facility adds dedicated computing capacity for local engineering and research workloads.",
            "Dense compute racks depend on coordinated cooling, networking, and power infrastructure at the site.",
            "Local developers gain closer access to managed infrastructure for controlled model training and inference work.",
            "Technicians commission racks, connect fiber, verify cooling, and maintain power delivery across the facility.",
            "Teams can test applications against dedicated hardware while keeping operations inside one managed environment.",
            "Useful results still depend on data, software, evaluation methods, availability, and disciplined operations.",
        ]
        return {
            "title": "Firebird Launches Armenia AI Factory",
            "narration": _FIXED_NARRATION,
            "scenes": [
                {"scene_id": index, "heading": f"Evidence Beat {index}", "body": body}
                for index, body in enumerate(bodies)
            ],
        }

    def test_final_boundary_repairs_internal_copy_and_preserves_evidence_metadata(self) -> None:
        from factory import canary, local_llm, source_attributed_llm

        source = self._source()
        bad = self._bad_package()
        fixed = self._focused_rewrite()
        self.assertEqual(140, len(_FIXED_NARRATION.split()))

        def bad_generate(_settings, _sources, _strategy):
            return bad

        def focused_chat(_settings, _prompt, *, attempts=3):
            self.assertEqual(1, attempts)
            return fixed

        def unexpected_package_parser(_settings, _sources, _raw):
            raise AssertionError("package parser not expected")

        def repair_prompt(*_args, **_kwargs):
            return "repair"

        with patch.object(source_attributed_llm, "generate_package", new=bad_generate), patch.object(
            canary,
            "generate_package",
            new=bad_generate,
        ), patch.object(local_llm, "_chat", new=focused_chat), patch.object(
            local_llm,
            "_package_from_raw",
            new=unexpected_package_parser,
        ), patch.object(local_llm, "_repair_prompt", new=repair_prompt):
            v66._install_consumer_copy_gate()
            repaired = canary.generate_package(SimpleNamespace(), [source], SimpleNamespace())

        self.assertEqual("Firebird Launches Armenia AI Factory", repaired.title)
        self.assertEqual(_FIXED_NARRATION, repaired.narration)
        self.assertEqual((), v66.consumer_editorial_failures_v66(repaired))
        self.assertEqual(bad.source_urls, repaired.source_urls)
        self.assertEqual(bad.source_publishers, repaired.source_publishers)
        self.assertEqual(
            [scene.source_index for scene in bad.scenes],
            [scene.source_index for scene in repaired.scenes],
        )
        self.assertEqual(
            [scene.visual for scene in bad.scenes],
            [scene.visual for scene in repaired.scenes],
        )

    def test_focused_rewrite_rejects_scene_reordering_or_missing_ids(self) -> None:
        bad = self._bad_package()
        raw = self._focused_rewrite()
        raw["scenes"] = list(raw["scenes"])[1:]
        with self.assertRaisesRegex(Exception, "incomplete copy object|scene ids"):
            v66._apply_focused_editorial_rewrite_v66(bad, raw)

    def test_package_prompt_detection_covers_initial_and_repair_prompts(self) -> None:
        self.assertTrue(v66._is_package_prompt("SOURCE ENTRIES:\nReturn one JSON object containing:"))
        self.assertTrue(v66._is_package_prompt("SOURCE ENTRIES:\nPREVIOUS JSON:"))
        self.assertFalse(v66._is_package_prompt("scene attribution only"))


if __name__ == "__main__":
    unittest.main()
