from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from factory.production_human_review_handoff_v61 import build_human_review_dossier_v61


class HumanReviewHandoffV61Tests(unittest.TestCase):
    def test_complete_temporal_bundle_is_ready_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "video.mp4").write_bytes(b"video")
            (root / "narration.wav").write_bytes(b"audio")
            shots = [
                {
                    "shot_id": index,
                    "start_seconds": index * 2.8,
                    "duration_seconds": 2.8,
                    "renderer": "wan_i2v",
                    "semantic_claim": f"claim {index}",
                }
                for index in range(20)
            ]
            assets = [
                {
                    "scene_index": index,
                    "media_type": "video",
                    "sha256": f"sha-{index}",
                }
                for index in range(20)
            ]
            payloads = {
                "visual-pipeline-manifest.json": {"shots": shots},
                "scene-media-manifest.json": {
                    "assets": assets,
                    "expected_temporal_shots": 20,
                    "realized_temporal_shots": 20,
                    "image_fallback_allowed": False,
                    "digital_zoom_motion_allowed": False,
                },
                "visual-composition-manifest.json": {
                    "shots": shots,
                    "scene_media": assets,
                    "source_asset_looping": False,
                },
                "video-qc-report.json": {"passed": True},
                "voice-review-manifest.json": {"metrics": {"estimated_wpm": 142.0}},
                "animated-captions.json": {"all_rendered_cues_fit": True},
                "vimax-plan.json": {"status": "planned", "shot_descriptions": [{}] * 20},
            }
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")

            dossier = build_human_review_dossier_v61(root)

            self.assertTrue(dossier["automated_precheck_passed"])
            self.assertEqual("awaiting_human_review", dossier["status"])
            self.assertEqual("blocked_pending_human_review", dossier["release_decision"])
            self.assertEqual(20, len(dossier["shot_review_samples"]))
            self.assertTrue(all(item["status"] == "pending_human" for item in dossier["human_checklist"]))

    def test_any_image_fallback_blocks_human_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "video.mp4").write_bytes(b"video")
            (root / "narration.wav").write_bytes(b"audio")
            shots = [{"shot_id": 0, "start_seconds": 0.0, "duration_seconds": 2.5}]
            assets = [{"scene_index": 0, "media_type": "image", "sha256": "sha-0"}]
            for name, payload in {
                "visual-pipeline-manifest.json": {"shots": shots},
                "scene-media-manifest.json": {
                    "assets": assets,
                    "expected_temporal_shots": 1,
                    "realized_temporal_shots": 0,
                    "image_fallback_allowed": False,
                    "digital_zoom_motion_allowed": False,
                },
                "visual-composition-manifest.json": {"shots": shots, "source_asset_looping": False},
                "video-qc-report.json": {"passed": True},
                "voice-review-manifest.json": {"metrics": {"estimated_wpm": 142.0}},
                "animated-captions.json": {"all_rendered_cues_fit": True},
                "vimax-plan.json": {"status": "planned"},
            }.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")

            dossier = build_human_review_dossier_v61(root)

            self.assertFalse(dossier["automated_precheck_passed"])
            self.assertEqual("blocked_before_human_review", dossier["status"])
            self.assertFalse(dossier["checks"]["all_source_media_temporal"])


if __name__ == "__main__":
    unittest.main()
