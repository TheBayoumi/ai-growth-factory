from __future__ import annotations

import unittest
from pathlib import Path

from factory.editorial_timeline import build_editorial_plan
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.video_profile import VideoProfile
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class EditorialAdaptiveCapacityV35Tests(unittest.TestCase):
    def _package(self) -> VideoPackage:
        scenes = [
            Scene(
                heading=f"Security beat {index}",
                body=f"Grounded security and governance claim {index}",
                visual=f"Physical cybersecurity collaboration setup {index}",
                source_index=0,
            )
            for index in range(6)
        ]
        return VideoPackage(
            topic="Open secure AI governance",
            narration="",
            title="Secure agentic AI guidelines",
            description="description",
            tags=["ai", "security"],
            thumbnail_text="Secure AI",
            top_comment="comment",
            scenes=scenes,
            source_urls=["https://example.com/source"],
            source_publishers=["Example"],
        )

    def _plan(self) -> VisualPlan:
        scenes = tuple(
            SceneVisualPrompt(
                scene_index=index,
                source_index=0,
                role=("hook", "evidence", "mechanism", "comparison", "implication", "cta")[index],
                generation_mode="image",
                image_prompt=f"Security documentary setup {index}",
                motion_prompt=f"Controlled motion {index}",
                negative_prompt="text, logo",
                continuity_anchor="neutral documentary lighting",
                caption_safe_zone="bottom",
                seed=500 + index,
                duration_seconds=4.0,
            )
            for index in range(6)
        )
        return VisualPlan(
            prompt_version="test-v35",
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="style only",
            image_model="test-image",
            video_model="test-video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="capacity-v35",
            scenes=scenes,
        )

    def test_real_eleven_beat_narration_uses_twenty_two_shots_inside_hard_capacity(self) -> None:
        texts = (
            "NVIDIA's Open Secure AI Alliance is developing new cybersecurity guidelines for agentic AI.",
            "The Linux Foundation has released a Request for Comments on Shared AI Findings Exchange, aiming to improve transparency and security.",
            "These guidelines come as Black Hat conference begins in Las Vegas.",
            "The initiative seeks to address growing concerns about AI safety and data protection.",
            "The alliance includes over 120 organizations working on standardized protocols.",
            "The goal is to create a framework that ensures secure and ethical AI practices.",
            "This effort reflects a broader push for accountability in AI development.",
            "The guidelines are part of a larger movement to safeguard digital systems.",
            "The initiative highlights the importance of collaboration in AI governance.",
            "The Linux Foundation is leading the effort to establish clear standards.",
            "Before adoption, read the linked source and test the claim on a controlled task.",
        )
        starts = (
            0.0,
            5.860333333333333,
            14.312208333333334,
            19.012541666666667,
            24.555875,
            29.145708333333335,
            35.094375,
            39.798291666666664,
            44.93029166666666,
            49.24829166666666,
            53.95866666666666,
        )
        total_duration = 59.76733333333333
        segments = tuple(
            NarrationSegment(
                segment_id=index,
                text=text,
                instruction="test",
                audio_path=Path(f"capacity-{index}.wav"),
                start_seconds=start,
                end_seconds=(starts[index + 1] - 0.14 if index + 1 < len(starts) else total_duration),
            )
            for index, (text, start) in enumerate(zip(texts, starts, strict=True))
        )
        profile = VideoProfile()
        expanded, shots = build_editorial_plan(
            plan=self._plan(),
            package=self._package(),
            segments=segments,
            total_duration=total_duration,
            profile=profile,
        )

        self.assertEqual(profile.target_shots, 20)
        self.assertEqual(profile.maximum_shots, 24)
        self.assertEqual(len(shots), 22)
        self.assertEqual(len(expanded.scenes), 22)
        self.assertGreaterEqual(len(shots), profile.target_shots)
        self.assertLessEqual(len(shots), profile.maximum_shots)
        self.assertEqual(sum(shot.renderer == "wan_i2v" for shot in shots), profile.wan_shots)
        self.assertGreaterEqual(sum(shot.start_seconds < 10.0 for shot in shots), 4)
        self.assertLessEqual(max(shot.duration_seconds for shot in shots), profile.maximum_shot_seconds)
        self.assertAlmostEqual(sum(shot.duration_seconds for shot in shots), total_duration, places=4)
        represented = {shot.segment_id for shot in shots}
        self.assertEqual(represented, set(range(len(segments))))

    def test_hard_capacity_remains_fail_closed(self) -> None:
        profile = VideoProfile(maximum_shots=20)
        profile.validate()
        texts = tuple(f"A deliberately long independent beat number {index} requires two shots." for index in range(11))
        starts = tuple(index * 5.0 for index in range(11))
        segments = tuple(
            NarrationSegment(
                segment_id=index,
                text=text,
                instruction="test",
                audio_path=Path(f"overflow-{index}.wav"),
                start_seconds=start,
                end_seconds=start + 4.9,
            )
            for index, (text, start) in enumerate(zip(texts, starts, strict=True))
        )
        with self.assertRaisesRegex(ValueError, "above configured maximum 20"):
            build_editorial_plan(
                plan=self._plan(),
                package=self._package(),
                segments=segments,
                total_duration=55.0,
                profile=profile,
            )


if __name__ == "__main__":
    unittest.main()
