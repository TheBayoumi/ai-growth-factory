from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.production_hitl_checkpoint_v71 import (
    HumanCheckpointApprovalError,
    _load_editorial_timeline,
    finalize_hitl_checkpoint_v71,
    record_hitl_decision_v71,
    require_approved_hitl_decision_v71,
    verify_hitl_checkpoint_v71,
)
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ProductionHitlCheckpointV71Tests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path, *, machine_passed: bool = True) -> None:
        for name in (
            "package.json",
            "voice-review-manifest.json",
            "visual-plan.json",
            "vimax-plan.json",
            "keyframe-manifest.json",
            "production-preflight.json",
            "editorial-timeline.json",
        ):
            (root / name).write_text(json.dumps({"name": name}), encoding="utf-8")
        (root / "narration.wav").write_bytes(b"approved-audio")
        keyframe_dir = root / "visual-keyframes"
        keyframe_dir.mkdir()
        shots = []
        records = []
        for index in range(20):
            name = f"scene-{index:02d}-keyframe.png"
            path = keyframe_dir / name
            path.write_bytes(f"frame-{index}".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            shots.append({"shot_id": index, "keyframe": name, "keyframe_sha256": digest})
            records.append(f"{index:04d}:{digest}")
        set_digest = hashlib.sha256("\n".join(records).encode()).hexdigest()
        (root / "keyframe-human-review-dossier.json").write_text(
            json.dumps(
                {
                    "status": "awaiting_human_keyframe_review",
                    "machine_keyframe_review_passed": machine_passed,
                    "machine_review_disposition": "passed" if machine_passed else "advisory_human_arbitration",
                    "expected_shots": 20,
                    "realized_keyframes": 20,
                    "keyframe_set_sha256": set_digest,
                    "shots": shots,
                }
            ),
            encoding="utf-8",
        )

    def test_checkpoint_digest_binds_code_files_and_all_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-test"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="a" * 40)
            verified = verify_hitl_checkpoint_v71(
                root,
                approved_keyframe_sha256=manifest["approval_subject_sha256"],
                approved_code_sha="a" * 40,
            )
            self.assertEqual(manifest["approval_subject_sha256"], verified["approval_subject_sha256"])
            dossier = json.loads((root / "keyframe-human-review-dossier.json").read_text())
            self.assertEqual(manifest["approval_subject_sha256"], dossier["approval_subject_sha256"])

    def test_human_can_seal_complete_advisory_machine_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-advisory"
            root.mkdir()
            self._fixture(root, machine_passed=False)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="e" * 40)
            self.assertFalse(manifest["machine_keyframe_review_passed"])
            self.assertEqual("advisory_human_arbitration", manifest["machine_review_disposition"])

    def test_temporal_generation_requires_explicit_human_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-decision"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="f" * 40)
            with self.assertRaisesRegex(HumanCheckpointApprovalError, "checkpoint JSON"):
                require_approved_hitl_decision_v71(
                    root,
                    approval_subject_sha256=manifest["approval_subject_sha256"],
                    code_sha="f" * 40,
                )

    def test_human_simulation_approval_binds_every_sealed_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-sim"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="1" * 40)
            decision = record_hitl_decision_v71(
                root,
                approval_subject_sha256=manifest["approval_subject_sha256"],
                code_sha="1" * 40,
                reviewer_kind="human_simulation",
                verdict="approve",
                reviewed_shot_ids=tuple(range(20)),
                notes=("Reviewed all 20 storyboard frames.",),
            )
            approval, verified = require_approved_hitl_decision_v71(
                root,
                approval_subject_sha256=manifest["approval_subject_sha256"],
                code_sha="1" * 40,
            )
            self.assertEqual(20, approval["expected_shots"])
            self.assertEqual("human_simulation", verified["reviewer_kind"])
            self.assertEqual("approve", verified["verdict"])
            self.assertEqual(list(range(20)), verified["reviewed_shot_ids"])
            self.assertEqual(decision["decision_sha256"], verified["decision_sha256"])

    def test_partial_simulated_approval_cannot_authorize_wan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-partial"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="2" * 40)
            with self.assertRaisesRegex(HumanCheckpointApprovalError, "every sealed keyframe"):
                record_hitl_decision_v71(
                    root,
                    approval_subject_sha256=manifest["approval_subject_sha256"],
                    code_sha="2" * 40,
                    reviewer_kind="human_simulation",
                    verdict="approve",
                    reviewed_shot_ids=tuple(range(19)),
                )

    def test_tampered_human_decision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-tampered-decision"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="3" * 40)
            record_hitl_decision_v71(
                root,
                approval_subject_sha256=manifest["approval_subject_sha256"],
                code_sha="3" * 40,
                reviewer_kind="human_simulation",
                verdict="approve",
                reviewed_shot_ids=tuple(range(20)),
            )
            path = root / "hitl-decision.json"
            decision = json.loads(path.read_text(encoding="utf-8"))
            decision["notes"] = ["changed after approval"]
            path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(HumanCheckpointApprovalError, "decision digest"):
                require_approved_hitl_decision_v71(
                    root,
                    approval_subject_sha256=manifest["approval_subject_sha256"],
                    code_sha="3" * 40,
                )

    @staticmethod
    def _expanded_plan_for_timeline() -> VisualPlan:
        scenes = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=index,
                role="evidence",
                generation_mode="wan_i2v",
                image_prompt=f"prompt {index}",
                motion_prompt=f"motion {index}",
                negative_prompt="text",
                continuity_anchor="same world",
                caption_safe_zone="lower",
                seed=100 + index,
                duration_seconds=2.0,
            )
            for index in range(2)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="same world",
            image_model="image",
            video_model="wan",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="a" * 64,
            scenes=scenes,
        )

    def test_exact_editorial_timeline_is_restored_without_reexpansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"target_shots": 2, "wan_shots": 2}
            payload = {
                "profile": profile,
                "duration_seconds": 4.0,
                "shot_count": 2,
                "shots": [
                    {
                        "shot_id": index,
                        "beat_index": index,
                        "segment_id": 0,
                        "package_scene_index": index,
                        "source_index": index,
                        "start_seconds": float(index * 2),
                        "duration_seconds": 2.0,
                        "renderer": "wan_i2v",
                        "semantic_claim": f"claim {index}",
                        "visual_direction": f"direction {index}",
                        "treatment": f"treatment {index}",
                        "seed": 100 + index,
                    }
                    for index in range(2)
                ],
            }
            (root / "editorial-timeline.json").write_text(json.dumps(payload), encoding="utf-8")
            fake_profile = SimpleNamespace(as_dict=lambda: profile)
            with patch("factory.video_profile.VideoProfile.from_env", return_value=fake_profile):
                shots = _load_editorial_timeline(
                    root, self._expanded_plan_for_timeline(), narration_duration=4.0
                )
            self.assertEqual([0, 1], [item.shot_id for item in shots])
            self.assertEqual([0.0, 2.0], [item.start_seconds for item in shots])
            self.assertTrue(all(item.renderer == "wan_i2v" for item in shots))

    def test_modified_keyframe_invalidates_human_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-test"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="b" * 40)
            (root / "visual-keyframes" / "scene-07-keyframe.png").write_bytes(b"changed-after-review")
            with self.assertRaisesRegex(HumanCheckpointApprovalError, "changed after review"):
                verify_hitl_checkpoint_v71(
                    root,
                    approved_keyframe_sha256=manifest["approval_subject_sha256"],
                    approved_code_sha="b" * 40,
                )

    def test_missing_or_wrong_approval_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "20260809T120000Z-test"
            root.mkdir()
            self._fixture(root)
            manifest = finalize_hitl_checkpoint_v71(root, code_sha="c" * 40)
            with self.assertRaises(HumanCheckpointApprovalError):
                verify_hitl_checkpoint_v71(root, approved_keyframe_sha256="", approved_code_sha="c" * 40)
            with self.assertRaises(HumanCheckpointApprovalError):
                verify_hitl_checkpoint_v71(
                    root,
                    approved_keyframe_sha256=manifest["approval_subject_sha256"],
                    approved_code_sha="d" * 40,
                )


if __name__ == "__main__":
    unittest.main()
