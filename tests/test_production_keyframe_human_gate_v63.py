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

    def test_complete_keyframe_set_waits_for_human_before_wan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_assets = []
            for index in range(20):
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

            dossier = build_keyframe_human_review_dossier_v63(
                plan=self._plan(), output_dir=root
            )

            self.assertEqual("awaiting_human_keyframe_review", dossier["status"])
            self.assertEqual("blocked_pending_human_keyframe_review", dossier["release_decision"])
            self.assertTrue(dossier["machine_keyframe_review_passed"])
            self.assertEqual(20, dossier["realized_keyframes"])
            self.assertTrue(all(item["generation_mode"] == "wan_i2v" for item in dossier["shots"]))
            self.assertTrue(all(item["review"]["accept_for_temporal_generation"] is None for item in dossier["shots"]))

    def test_machine_failure_is_still_packaged_for_human_diagnosis(self) -> None:
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
            self.assertEqual(1, dossier["realized_keyframes"])
            self.assertIn("semantic review", dossier["machine_error"])


if __name__ == "__main__":
    unittest.main()
