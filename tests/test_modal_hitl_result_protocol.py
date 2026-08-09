from __future__ import annotations

import unittest
from pathlib import Path


class ModalHitlResultProtocolTests(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    def test_preproduction_streams_structured_result_without_registering_second_entrypoint(self) -> None:
        source = (self._repo_root() / "cloud" / "modal_vimax_hitl_preproduction.py").read_text(encoding="utf-8")
        self.assertNotIn('@app.local_entrypoint()', source)
        self.assertIn('HITL_RESULT_JSON=', source)
        self.assertIn('_emit_gate1_result(final)', source)
        self.assertIn('"hitl-results"', source)
        self.assertIn('gate1-result.json', source)

    def test_manual_hitl_workflow_exports_before_classifying_human_gate(self) -> None:
        source = (self._repo_root() / ".github" / "workflows" / "vimax-remotion-canary.yml").read_text(encoding="utf-8")
        self.assertIn('modal run -m cloud.modal_vimax_hitl_preproduction::app.prepare_vimax_keyframe_review --code-sha "$CODE_SHA"', source)
        self.assertIn('HITL_RESULT_JSON=', source)
        self.assertIn('hitl-results/${CODE_SHA}.json', source)
        upload = source.index('uses: actions/upload-artifact@v4')
        enforce = source.index('name: Enforce human/release gate after evidence export')
        self.assertLess(upload, enforce)
        self.assertNotIn("grep -q 'awaiting_human_keyframe_review'", source)


if __name__ == "__main__":
    unittest.main()
