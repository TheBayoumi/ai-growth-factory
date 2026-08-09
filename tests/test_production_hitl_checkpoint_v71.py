from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from factory.production_hitl_checkpoint_v71 import (
    HumanCheckpointApprovalError,
    finalize_hitl_checkpoint_v71,
    verify_hitl_checkpoint_v71,
)


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
