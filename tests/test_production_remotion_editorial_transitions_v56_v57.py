from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_remotion_editorial_transitions_v56 import (
    annotate_remotion_story_beats_v56,
)
from factory.production_remotion_transition_policy_v57 import (
    build_remotion_editorial_transition_evidence_v57,
)
from factory.remotion_contract import CameraSpec, RemotionRenderSpec, RenderShot


class RemotionEditorialTransitionV56V57Tests(unittest.TestCase):
    @staticmethod
    def _render_shot(shot_id: int, start_frame: int) -> RenderShot:
        return RenderShot(
            shot_id=shot_id,
            start_frame=start_frame,
            duration_in_frames=90,
            semantic_claim=f"claim {shot_id}",
            purpose="continuity shot",
            first_frame_prompt=f"first {shot_id}",
            last_frame_prompt=f"last {shot_id}",
            motion_prompt="controlled temporal motion",
            camera=CameraSpec("medium", "eye_level", "dolly_in"),
            renderer="video_clip",
            media_path=f"shot-{shot_id}.mp4",
            keyframe_path=f"shot-{shot_id}.png",
            source_index=0,
            seed=shot_id + 1,
        )

    def test_v56_preserves_package_scene_identity_in_render_contract(self) -> None:
        spec = RemotionRenderSpec(
            schema_version="vimax-remotion-v1",
            width=1080,
            height=1920,
            fps=30,
            duration_in_frames=180,
            audio_path="narration.wav",
            background_music_path=None,
            title="test",
            source_label="source",
            shots=(self._render_shot(0, 0), self._render_shot(1, 90)),
            captions=(),
            transition_frames=5,
        )
        source_shots = [
            SimpleNamespace(package_scene_index=0),
            SimpleNamespace(package_scene_index=1),
        ]

        updated = annotate_remotion_story_beats_v56(spec, source_shots)

        self.assertTrue(updated.shots[0].purpose.startswith("package_scene:0;"))
        self.assertTrue(updated.shots[1].purpose.startswith("package_scene:1;"))
        self.assertEqual(180, updated.duration_in_frames)

    @staticmethod
    def _payload_shot(shot_id: int, package_scene: int) -> dict[str, object]:
        return {
            "shot_id": shot_id,
            "start_frame": shot_id * 90,
            "duration_in_frames": 90,
            "renderer": "video_clip",
            "purpose": f"package_scene:{package_scene}; continuity shot",
        }

    def test_v57_dissolves_only_between_story_beats(self) -> None:
        evidence = build_remotion_editorial_transition_evidence_v57(
            {
                "transition_frames": 5,
                "shots": [
                    self._payload_shot(0, 0),
                    self._payload_shot(1, 0),
                    self._payload_shot(2, 1),
                    self._payload_shot(3, 1),
                    self._payload_shot(4, 2),
                ],
            }
        )

        self.assertEqual(4, evidence["boundary_count"])
        self.assertEqual(2, evidence["realized_transition_count"])
        self.assertEqual(2, evidence["hard_cut_count"])
        self.assertEqual(
            [(item["outgoing_shot_id"], item["incoming_shot_id"]) for item in evidence["transitions"]],
            [(1, 2), (3, 4)],
        )
        self.assertTrue(
            all(item["reason"] == "story_beat_change" for item in evidence["transitions"])
        )
        self.assertTrue(
            all(item["reason"] == "continuity_cut_same_story_beat" for item in evidence["hard_cuts"])
        )

    def test_v57_still_rejects_all_hard_cut_plan_with_no_beat_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "required at least 2"):
            build_remotion_editorial_transition_evidence_v57(
                {
                    "transition_frames": 5,
                    "shots": [
                        self._payload_shot(0, 0),
                        self._payload_shot(1, 0),
                        self._payload_shot(2, 0),
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
