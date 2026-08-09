from __future__ import annotations

import unittest

from factory.models import Scene, VideoPackage
from factory.production_vimax_focused_copy_protocol_v69 import (
    apply_focused_narration_rewrite_v69,
    focused_narration_prompt_v69,
    install_production_vimax_focused_copy_protocol_v69,
)


class ProductionViMaxFocusedCopyProtocolV69Tests(unittest.TestCase):
    @staticmethod
    def _package() -> VideoPackage:
        return VideoPackage(
            topic="Firebird AI factory",
            narration="Internal source process language that requires repair.",
            title="Firebird Launches Armenia AI Factory",
            description="Description.",
            tags=["AI", "Armenia", "compute", "infrastructure", "engineering", "data center", "research", "deployment"],
            thumbnail_text="AI FACTORY",
            top_comment="What matters most?",
            scenes=[
                Scene(
                    heading=f"Beat {index}",
                    body=f"Source-backed infrastructure evidence beat {index}.",
                    visual=f"Immutable visual {index}.",
                    source_index=0,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/source"],
            source_publishers=["Publisher"],
        )

    def test_narration_only_rewrite_preserves_every_other_package_field(self) -> None:
        package = self._package()
        narration = "A repaired audience-facing narration keeps all useful source-grounded facts and removes internal process language."
        repaired = apply_focused_narration_rewrite_v69(package, {"narration": narration})
        self.assertEqual(narration, repaired.narration)
        self.assertEqual(package.title, repaired.title)
        self.assertEqual(package.scenes, repaired.scenes)
        self.assertEqual(package.source_urls, repaired.source_urls)
        self.assertEqual(package.source_publishers, repaired.source_publishers)

    def test_forbidden_extra_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "forbidden fields"):
            apply_focused_narration_rewrite_v69(
                self._package(),
                {"narration": "Narration.", "title": "Unauthorized title rewrite"},
            )

    def test_missing_narration_reports_response_keys(self) -> None:
        with self.assertRaisesRegex(Exception, "response_keys"):
            apply_focused_narration_rewrite_v69(self._package(), {"result": "text"})

    def test_installer_declares_scene_copy_fallback_to_full_v66_schema(self) -> None:
        import inspect

        source = inspect.getsource(install_production_vimax_focused_copy_protocol_v69)
        self.assertIn('"scene copy"', source)
        self.assertIn("full_prompt", source)
        self.assertIn("full_apply", source)

    def test_prompt_requests_one_field_only(self) -> None:
        package = self._package()
        source = type(
            "Source",
            (),
            {
                "url": "https://example.com/source",
                "publisher": "Publisher",
                "author": "Publisher",
                "authority": "Publisher",
                "title": "Publisher launches infrastructure",
                "summary": "Concrete source evidence about infrastructure and engineering capacity.",
            },
        )()
        prompt = focused_narration_prompt_v69(package, [source], "bad internal copy")
        self.assertIn('Return exactly one field: narration', prompt)
        self.assertIn('{"narration":"..."}', prompt)
        self.assertNotIn('"scenes": [', prompt)


if __name__ == "__main__":
    unittest.main()
