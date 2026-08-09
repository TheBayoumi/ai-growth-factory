from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from factory.production_keyframe_human_gate_v63 import (
    build_keyframe_human_review_dossier_v63,
)


class KeyframeHumanGateV63Tests(unittest.TestCase):
    @staticmethod
    def _plan(count: int = 20) -> SimpleNamespace:
        return SimpleNamespace(
            prompt_version="vimax-script2video@test",
            scenes=tuple(
                SimpleNamespace(
                    scene_index=index,
                    image_prompt=f"physical infrastructure direction {index}",
                    motion_prompt=f"temporal motion {index}",
                    duration_seconds=2.9,
                    generation_mode="wan_i2v",
                )
                for index in range(count)
            ),
        )

    @staticmethod
    def _write_keyframes(root: Path, count: int) -> None:
        manifest_assets = []
        for index in range(count):
            path = root / f"scene-{index:02d}-keyframe.png"
            Image.new("RGB", (64, 96), (index * 7 % 255, 50, 100)).save(path)
            manifest_assets.append(
                {
                    "scene_index": index,
                    "model": "test-model",
                    "prompt": f"compiled infrastructure prompt {index}",
                }
            )
        (root / "keyframe-manifest.json").write_text(
            json.dumps({"assets": manifest_assets}), encoding="utf-8"
        )

    def test_complete_keyframe_set_waits_for_human_before_wan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_keyframes(root, 20)

            dossier = build_keyframe_human_review_dossier_v63(
                plan=self._plan(), output_dir=root
            )

            self.assertEqual("awaiting_human_keyframe_review", dossier["status"])
            self.assertEqual("blocked_pending_human_keyframe_review", dossier["release_decision"])
            self.assertTrue(dossier["machine_keyframe_review_passed"])
            self.assertEqual("passed", dossier["machine_review_disposition"])
            self.assertEqual(20, dossier["realized_keyframes"])
            self.assertEqual(64, len(dossier["keyframe_set_sha256"]))
            self.assertTrue(all(item["generation_mode"] == "wan_i2v" for item in dossier["shots"]))
            self.assertTrue(all(item["review"]["accept_for_temporal_generation"] is None for item in dossier["shots"]))

    def test_incomplete_machine_failure_is_still_fail_closed_for_human_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "scene-00-keyframe.png"
            Image.new("RGB", (64, 96), (25, 50, 75)).save(path)
            dossier = build_keyframe_human_review_dossier_v63(
                plan=self._plan(2),
                output_dir=root,
                machine_error="scene 1 failed semantic review",
            )
            self.assertEqual("blocked_machine_keyframe_review", dossier["status"])
            self.assertFalse(dossier["machine_keyframe_review_passed"])
            self.assertEqual("blocked_incomplete_evidence", dossier["machine_review_disposition"])
            self.assertEqual(1, dossier["realized_keyframes"])
            self.assertIsNone(dossier["keyframe_set_sha256"])
            self.assertIn("semantic review", dossier["machine_error"])

    def test_complete_machine_semantic_failure_requires_human_arbitration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_keyframes(root, 2)
            dossier = build_keyframe_human_review_dossier_v63(
                plan=self._plan(2),
                output_dir=root,
                machine_error="scene 1 failed semantic review",
            )
            self.assertEqual("awaiting_human_keyframe_review", dossier["status"])
            self.assertFalse(dossier["machine_keyframe_review_passed"])
            self.assertEqual("advisory_human_arbitration", dossier["machine_review_disposition"])
            self.assertEqual(2, dossier["realized_keyframes"])
            self.assertEqual(64, len(dossier["keyframe_set_sha256"]))
            self.assertTrue(dossier["human_review_required"])


if __name__ == "__main__":
    unittest.main()
