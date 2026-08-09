from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.production_hitl_checkpoint_v71 import (
    finalize_hitl_checkpoint_v71,
    record_hitl_decision_v71,
)
from factory.production_hitl_repair_v72 import HitlRepairSeedError, load_hitl_repair_seed_v72


class ProductionHitlRepairV72Tests(unittest.TestCase):
    @staticmethod
    def _director(index: int, direction: str) -> str:
        return (
            f"[VIMAX_SHOT_INDEX={index}] Factual technology documentary shot synchronized to this exact spoken sentence: Claim {index}. "
            f"Supporting source-grounded visual direction: {direction}. "
            f"Shot treatment: medium eye-level documentary framing. ViMax first frame: {direction}."
        )

    def _checkpoint(self, root: Path) -> dict[str, object]:
        directions = ["technician connects fiber to a rack", "engineer checks cooling manifold", "worker rolls a server cabinet"]
        keyframe_dir = root / "visual-keyframes"
        keyframe_dir.mkdir()
        assets = []
        shots = []
        records = []
        for index, direction in enumerate(directions):
            name = f"scene-{index:02d}-keyframe.png"
            path = keyframe_dir / name
            path.write_bytes((f"sealed-frame-{index}-" * 100).encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            director = self._director(index, direction)
            assets.append(
                {
                    "scene_index": index,
                    "model": "test-model",
                    "seed": 100 + index,
                    "width": 704,
                    "height": 1280,
                    "sha256": digest,
                    "entropy": 7.0,
                    "prompt": f"compiled {direction}",
                    "negative_prompt": "text logo",
                    "director_prompt": director,
                    "prompt_word_count": 10,
                    "prompt_word_budget": 58,
                    "prompt_compiler_version": "visual-compiler-v52-vimax-authority",
                    "caption_zone_detail_before": 8.0,
                    "caption_zone_detail_after": 4.0,
                    "caption_zone_repaired": True,
                }
            )
            shots.append({"shot_id": index, "keyframe": name, "keyframe_sha256": digest})
            records.append(f"{index:04d}:{digest}")
        (root / "keyframe-manifest.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
        for name in (
            "package.json",
            "voice-review-manifest.json",
            "visual-plan.json",
            "vimax-plan.json",
            "production-preflight.json",
            "editorial-timeline.json",
        ):
            (root / name).write_text(json.dumps({"name": name}), encoding="utf-8")
        (root / "narration.wav").write_bytes(b"sealed-audio")
        set_digest = hashlib.sha256("\n".join(records).encode()).hexdigest()
        (root / "keyframe-human-review-dossier.json").write_text(
            json.dumps(
                {
                    "status": "awaiting_human_keyframe_review",
                    "machine_keyframe_review_passed": False,
                    "machine_review_disposition": "advisory_human_arbitration",
                    "expected_shots": 3,
                    "realized_keyframes": 3,
                    "keyframe_set_sha256": set_digest,
                    "shots": shots,
                }
            ),
            encoding="utf-8",
        )
        manifest = finalize_hitl_checkpoint_v71(root, code_sha="a" * 40)
        record_hitl_decision_v71(
            root,
            approval_subject_sha256=str(manifest["approval_subject_sha256"]),
            code_sha="a" * 40,
            reviewer_kind="human_simulation",
            verdict="reject",
            reviewed_shot_ids=(0, 1, 2),
            notes=("Shot 2 rejected by HITL.",),
        )
        return manifest

    def test_reuses_only_nonrejected_unchanged_direction_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source-canary"
            root.mkdir()
            manifest = self._checkpoint(root)
            current_scenes = {
                0: SimpleNamespace(image_prompt=self._director(0, "technician connects fiber to a rack")),
                1: SimpleNamespace(image_prompt=self._director(1, "engineer inspects exterior cooling equipment")),
                2: SimpleNamespace(image_prompt=self._director(2, "worker rolls a server cabinet")),
            }
            plan = SimpleNamespace(scenes=tuple(current_scenes.values()), width=704, height=1280)
            env = {
                "HITL_REPAIR_SOURCE_DIR": str(root),
                "HITL_REPAIR_APPROVAL_SUBJECT_SHA256": str(manifest["approval_subject_sha256"]),
                "HITL_REPAIR_SOURCE_CODE_SHA": "a" * 40,
                "HITL_REPAIR_REJECTED_SHOTS": "2",
            }
            with patch.dict("os.environ", env, clear=False):
                candidates, info = load_hitl_repair_seed_v72(
                    plan=plan,
                    current_scenes=current_scenes,
                    output_dir=Path(tmp) / "new-keyframes",
                )
            self.assertEqual([0], sorted(candidates))
            self.assertEqual([0], info["eligible_shots"])
            self.assertEqual([1], info["direction_changed_shots"])
            self.assertEqual([2], info["rejected_shots"])
            self.assertEqual(candidates[0].sha256, manifest["keyframes"]["scene-00-keyframe.png"])

    def test_repair_requires_explicit_rejected_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source-canary"
            root.mkdir()
            manifest = self._checkpoint(root)
            decision_path = root / "hitl-decision.json"
            decision = json.loads(decision_path.read_text())
            decision["verdict"] = "approve"
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            scene = SimpleNamespace(image_prompt=self._director(0, "technician connects fiber to a rack"))
            env = {
                "HITL_REPAIR_SOURCE_DIR": str(root),
                "HITL_REPAIR_APPROVAL_SUBJECT_SHA256": str(manifest["approval_subject_sha256"]),
                "HITL_REPAIR_SOURCE_CODE_SHA": "a" * 40,
                "HITL_REPAIR_REJECTED_SHOTS": "2",
            }
            with patch.dict("os.environ", env, clear=False):
                with self.assertRaisesRegex(HitlRepairSeedError, "decision digest"):
                    load_hitl_repair_seed_v72(
                        plan=SimpleNamespace(scenes=(scene, scene, scene), width=704, height=1280),
                        current_scenes={0: scene, 1: scene, 2: scene},
                        output_dir=Path(tmp) / "new-keyframes",
                    )

    def test_no_request_is_zero_cost_noop(self) -> None:
        with patch.dict(
            "os.environ",
            {"HITL_REPAIR_SOURCE_DIR": ""},
            clear=False,
        ):
            candidates, info = load_hitl_repair_seed_v72(
                plan=SimpleNamespace(scenes=(), width=704, height=1280),
                current_scenes={},
                output_dir=Path("unused"),
            )
        self.assertEqual({}, candidates)
        self.assertEqual({"enabled": False}, info)


if __name__ == "__main__":
    unittest.main()
