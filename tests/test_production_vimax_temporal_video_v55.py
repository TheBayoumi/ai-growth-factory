from __future__ import annotations

import unittest

from factory.production_vimax_temporal_video_v55 import (
    ensure_temporal_vimax_plan_v55,
    frame_num_for_duration_v55,
    native_temporal_media_failures_v55,
    temporal_profile_v55,
)
from factory.video_profile import VideoProfile
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class ViMaxTemporalVideoV55Tests(unittest.TestCase):
    @staticmethod
    def _plan() -> VisualPlan:
        scenes = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=0,
                role="hook" if index == 0 else "evidence",
                generation_mode="image" if index else "wan_i2v",
                image_prompt=f"Source-grounded shot {index} of a developer workflow.",
                motion_prompt=(
                    "Static camera. The subject remains in the same position with no significant changes."
                    if index % 2 == 0
                    else "Controlled pan right as the developer completes the depicted workflow."
                ),
                negative_prompt="text, logo, shake",
                continuity_anchor="same environment and subject identity",
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=index + 1,
                duration_seconds=2.75,
            )
            for index in range(20)
        )
        return VisualPlan(
            prompt_version="vimax-script2video@test",
            global_style="photorealistic documentary",
            palette="neutral graphite",
            lighting="natural documentary lighting",
            continuity_bible="preserve recurring subjects",
            image_model="image-model",
            video_model="video-model",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="abc",
            scenes=scenes,
        )

    def test_vimax_profile_requires_all_target_shots_as_temporal(self) -> None:
        profile = temporal_profile_v55(VideoProfile())
        self.assertEqual(profile.target_shots, profile.wan_shots)
        self.assertEqual(20, profile.wan_shots)
        self.assertEqual(profile.maximum_wan_shot_seconds, profile.maximum_shot_seconds)
        self.assertEqual(3.30, profile.maximum_shot_seconds)

    def test_every_vimax_scene_becomes_real_temporal_media(self) -> None:
        plan = ensure_temporal_vimax_plan_v55(self._plan())
        self.assertEqual(20, len(plan.scenes))
        self.assertTrue(all(scene.generation_mode == "wan_i2v" for scene in plan.scenes))
        self.assertTrue(
            all("native temporal-generation requirement" in scene.motion_prompt.casefold() for scene in plan.scenes)
        )
        self.assertNotIn("static camera", plan.scenes[0].motion_prompt.casefold())
        self.assertIn("controlled pan right", plan.scenes[1].motion_prompt.casefold())

    def test_dynamic_frame_count_covers_short_and_long_shots(self) -> None:
        self.assertEqual(41, frame_num_for_duration_v55(1.65, 24))
        self.assertEqual(81, frame_num_for_duration_v55(3.30, 24))
        with self.assertRaisesRegex(RuntimeError, "above configured maximum"):
            frame_num_for_duration_v55(4.25, 24)

    def test_image_fallback_is_release_blocking(self) -> None:
        failures = native_temporal_media_failures_v55(("video", "image", "video"))
        self.assertEqual(1, len(failures))
        self.assertIn("shot 1", failures[0])
        self.assertIn("slideshow/image fallback is forbidden", failures[0])

    def test_all_video_source_media_passes_native_media_gate(self) -> None:
        self.assertEqual((), native_temporal_media_failures_v55(("video",) * 20))


if __name__ == "__main__":
    unittest.main()
