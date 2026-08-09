from __future__ import annotations

import unittest
from pathlib import Path

from factory.models import NarrationSegment, Scene, VideoPackage
from factory.video_profile import VideoProfile
from factory.vimax_planner import (
    VIMAX_PROMPT_VERSION,
    ViMaxPlanningError,
    ViMaxShot,
    _balanced_frames,
    build_vimax_editorial_plan,
)
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ViMaxPlannerTests(unittest.TestCase):
    def test_shot_schema_is_strict_and_contiguous(self) -> None:
        shot = ViMaxShot.from_dict(
            {
                "idx": 0,
                "cam_idx": 1,
                "visual_desc": "Wide laboratory view",
                "ff_desc": "Wide first frame",
                "lf_desc": "Medium final frame",
                "motion_desc": "Slow dolly in while the engineer connects the cable",
                "variation_type": "medium",
                "variation_reason": "Composition changes without a cut",
                "audio_desc": "",
            },
            expected_idx=0,
        )
        self.assertEqual("medium", shot.variation_type)
        with self.assertRaises(ViMaxPlanningError):
            ViMaxShot.from_dict({"idx": 2, "cam_idx": 0}, expected_idx=0)

    def test_duration_allocator_frontloads_four_shots(self) -> None:
        profile = VideoProfile()
        frames = _balanced_frames(58 * 30, 16, profile, 30)
        self.assertEqual(58 * 30, sum(frames))
        self.assertLess(sum(frames[:3]), 10 * 30)
        self.assertTrue(
            all(
                round(profile.minimum_shot_seconds * 30) <= value
                <= round(profile.maximum_shot_seconds * 30)
                for value in frames
            )
        )

    def test_builds_exact_factory_timeline_from_vimax_shots(self) -> None:
        profile = VideoProfile()
        scenes = []
        wan = {0, 7, 14}
        for index in range(20):
            scenes.append(
                SceneVisualPrompt(
                    scene_index=index,
                    source_index=index % 2,
                    role="hook" if index == 0 else "cta" if index == 19 else "evidence",
                    generation_mode="wan_i2v" if index in wan else "image",
                    image_prompt=f"Concrete first frame for shot {index}",
                    motion_prompt=f"Controlled motion for shot {index}",
                    negative_prompt="text, logos",
                    continuity_anchor=f"camera {index % 4}; final frame {index}",
                    caption_safe_zone="lower_20_percent_overlay_only",
                    seed=index,
                    duration_seconds=2.9,
                )
            )
        plan = VisualPlan(
            prompt_version=VIMAX_PROMPT_VERSION,
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="pinned test plan",
            image_model="image",
            video_model="video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="abc",
            scenes=tuple(scenes),
        )
        package = VideoPackage(
            topic="topic",
            narration="narration",
            title="title",
            description="description",
            tags=[],
            thumbnail_text="thumb",
            top_comment="comment",
            scenes=[
                Scene("A", "A", "A", 0),
                Scene("B", "B", "B", 1),
                Scene("C", "C", "C", 0),
                Scene("D", "D", "D", 1),
                Scene("E", "E", "E", 0),
                Scene("F", "F", "F", 1),
            ],
            source_urls=["https://a", "https://b"],
            source_publishers=["A", "B"],
        )
        segments = tuple(
            NarrationSegment(
                segment_id=index,
                text=f"Narration claim {index}",
                instruction="",
                audio_path=Path("unused.wav"),
                start_seconds=index * (58 / 6),
                end_seconds=(index + 1) * (58 / 6),
            )
            for index in range(6)
        )
        expanded, shots = build_vimax_editorial_plan(
            plan=plan,
            package=package,
            segments=segments,
            total_duration=58.0,
            profile=profile,
        )
        self.assertEqual(20, len(shots))
        self.assertAlmostEqual(58.0, sum(item.duration_seconds for item in shots), places=5)
        self.assertEqual(3, sum(item.renderer == "wan_i2v" for item in shots))
        self.assertGreaterEqual(sum(item.start_seconds < 10.0 for item in shots), 4)
        self.assertEqual(20, len(expanded.scenes))


if __name__ == "__main__":
    unittest.main()
