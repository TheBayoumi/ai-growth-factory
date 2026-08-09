from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from factory.production_transition_evidence_v48 import (
    build_remotion_transition_evidence_v48,
    persist_remotion_transition_evidence_v48,
)


class ProductionTransitionEvidenceV48Tests(unittest.TestCase):
    @staticmethod
    def _shot(
        shot_id: int,
        renderer: str,
        start_frame: int,
        duration_in_frames: int = 90,
    ) -> dict[str, object]:
        return {
            "shot_id": shot_id,
            "renderer": renderer,
            "start_frame": start_frame,
            "duration_in_frames": duration_in_frames,
        }

    def test_evidence_mirrors_image_exit_crossfades_and_video_hard_cuts(self) -> None:
        evidence = build_remotion_transition_evidence_v48(
            {
                "transition_frames": 5,
                "shots": [
                    self._shot(0, "image_motion", 0),
                    self._shot(1, "image_motion", 90),
                    self._shot(2, "video_clip", 180),
                    self._shot(3, "image_motion", 270),
                    self._shot(4, "image_motion", 360),
                ],
            }
        )

        self.assertEqual(evidence["boundary_count"], 4)
        self.assertEqual(evidence["realized_transition_count"], 3)
        self.assertEqual(evidence["hard_cut_count"], 1)
        self.assertEqual(
            [item["outgoing_shot_id"] for item in evidence["transitions"]],
            [0, 1, 3],
        )
        self.assertEqual(evidence["hard_cuts"][0]["reason"], "video_clip_exit")

    def test_transition_duration_is_symmetric_across_both_shots(self) -> None:
        evidence = build_remotion_transition_evidence_v48(
            {
                "transition_frames": 5,
                "shots": [
                    self._shot(0, "image_motion", 0, 9),
                    self._shot(1, "image_motion", 9, 6),
                    self._shot(2, "video_clip", 15, 90),
                ],
            }
        )

        self.assertEqual(
            [item["duration_in_frames"] for item in evidence["transitions"]],
            [2, 2],
        )

    def test_rejects_a_plan_without_minimum_realized_transitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "required at least 2"):
            build_remotion_transition_evidence_v48(
                {
                    "transition_frames": 5,
                    "shots": [
                        self._shot(0, "video_clip", 0),
                        self._shot(1, "video_clip", 90),
                        self._shot(2, "image_motion", 180),
                    ],
                }
            )

    def test_persists_auditable_manifest_and_backend_specific_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            (workdir / "remotion-render-spec.json").write_text(
                json.dumps(
                    {
                        "transition_frames": 5,
                        "shots": [
                            self._shot(0, "image_motion", 0),
                            self._shot(1, "image_motion", 90),
                            self._shot(2, "video_clip", 180),
                            self._shot(3, "image_motion", 270),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (workdir / "visual-composition-manifest.json").write_text(
                json.dumps(
                    {
                        "renderer": "remotion_editorial_timeline_v45",
                        "shot_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            (workdir / "visual-compositor.log").write_text(
                "rendered by Remotion\n",
                encoding="utf-8",
            )

            evidence = persist_remotion_transition_evidence_v48(workdir)
            manifest = json.loads(
                (workdir / "visual-composition-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            log = (workdir / "visual-compositor.log").read_text(encoding="utf-8")

        self.assertIsNotNone(evidence)
        self.assertEqual(manifest["transition_count"], 2)
        self.assertEqual(
            manifest["transition_evidence"]["realized_transition_count"],
            2,
        )
        self.assertEqual(log.count("remotion_transition=opacity_crossfade"), 2)
        self.assertEqual(log.count("remotion_transition=hard_cut"), 1)
        self.assertNotIn("xfade=transition=fade", log)


if __name__ == "__main__":
    unittest.main()
