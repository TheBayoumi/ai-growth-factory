from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from factory.editorial_timeline import _wan_indices, build_editorial_plan
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.video_profile import VideoProfile
from factory.visual_prompt import SceneVisualPrompt, VisualPlan


class EditorialTimelineTests(unittest.TestCase):
    def _package(self) -> VideoPackage:
        scenes = [
            Scene(
                heading=f"Beat {index}",
                body=f"Concrete factual claim {index} about researchers and shared infrastructure",
                visual=f"Researchers use workspace tools for process {index}",
                source_index=0,
            )
            for index in range(6)
        ]
        return VideoPackage(
            topic="AI research infrastructure",
            narration=" ".join(scene.body for scene in scenes),
            title="A concrete research workflow",
            description="description",
            tags=["ai"],
            thumbnail_text="Research workflow",
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
                generation_mode="wan_i2v" if index in {0, 2, 3} else "image",
                image_prompt=f"Original visual prompt {index} for researchers and workspace tools",
                motion_prompt=f"Original motion prompt {index} with stable movement through the workflow",
                negative_prompt="text, logo",
                continuity_anchor="blue accent",
                caption_safe_zone="bottom",
                seed=100 + index,
                duration_seconds=4.0,
            )
            for index in range(6)
        )
        return VisualPlan(
            prompt_version="test",
            global_style="documentary",
            palette="neutral",
            lighting="natural",
            continuity_bible="style only",
            image_model="test-image",
            video_model="test-video",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="abc",
            scenes=scenes,
        )

    def _segments(self) -> tuple[NarrationSegment, ...]:
        starts = [0.0, 10.976, 23.096, 34.842, 41.155, 50.710]
        ends = starts[1:] + [56.147]
        return tuple(
            NarrationSegment(
                segment_id=index,
                text=f"Narration beat {index} with concrete factual words and context",
                instruction="test",
                audio_path=Path(f"segment-{index}.wav"),
                start_seconds=start,
                end_seconds=end,
            )
            for index, (start, end) in enumerate(zip(starts, ends, strict=True))
        )

    def test_v27_timing_expands_to_unique_short_shots(self) -> None:
        profile = VideoProfile()
        expanded, shots = build_editorial_plan(
            plan=self._plan(),
            package=self._package(),
            segments=self._segments(),
            total_duration=56.147,
            profile=profile,
        )
        self.assertGreaterEqual(len(shots), profile.minimum_shots)
        self.assertLessEqual(len(shots), profile.maximum_shots)
        self.assertGreaterEqual(
            sum(shot.start_seconds < 10.0 for shot in shots),
            profile.first_ten_seconds_minimum_shots,
        )
        self.assertEqual(sum(shot.renderer == "wan_i2v" for shot in shots), profile.wan_shots)
        self.assertTrue(all(shot.duration_seconds <= profile.maximum_shot_seconds for shot in shots))
        self.assertAlmostEqual(sum(shot.duration_seconds for shot in shots), 56.147, places=4)
        self.assertEqual(len({scene.seed for scene in expanded.scenes}), len(expanded.scenes))
        self.assertTrue(
            all("synchronized to this exact spoken sentence" in scene.image_prompt for scene in expanded.scenes)
        )
        self.assertTrue(all("Narration beat" in scene.image_prompt for scene in expanded.scenes))
        self.assertTrue(all("generic corridors" in scene.image_prompt for scene in expanded.scenes))

    def test_each_shot_maps_to_one_spoken_beat(self) -> None:
        _, shots = build_editorial_plan(
            plan=self._plan(),
            package=self._package(),
            segments=self._segments(),
            total_duration=56.147,
            profile=VideoProfile(),
        )
        self.assertEqual([shot.shot_id for shot in shots], list(range(len(shots))))
        self.assertTrue(all(0 <= shot.beat_index < 6 for shot in shots))
        self.assertTrue(all(0 <= shot.package_scene_index < 6 for shot in shots))
        self.assertTrue(all(shot.semantic_claim.startswith("Narration beat") for shot in shots))
        self.assertEqual(len({shot.shot_id for shot in shots}), len(shots))

    def test_wan_allocator_supports_more_than_three_configured_shots(self) -> None:
        _, shots = build_editorial_plan(
            plan=self._plan(),
            package=self._package(),
            segments=self._segments(),
            total_duration=56.147,
            profile=VideoProfile(),
        )
        selected = _wan_indices(shots, replace(VideoProfile(), wan_shots=6))

        self.assertEqual(len(selected), 6)
        self.assertIn(0, selected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_real_canary_duration_uses_active_sentence_aligned_shot_profile(self) -> None:
        package = self._package()
        texts = [
            "Microsoft Research launched Orchard as an open framework. Infrastructure reuse helps smaller models achieve strong performance.",
            "The framework supports scalable agentic systems and researcher collaboration. Reducing complexity helps teams evaluate diverse tasks.",
            "The research community can use and build upon the framework.",
            "Orchard focuses on scalability and measured performance.",
            "Researchers can share insights and improve AI systems.",
            "The community can contribute to ongoing development.",
        ]
        starts = [0.0, 14.862028, 32.124574, 40.956037, 48.289126, 54.967772]
        ends = starts[1:] + [61.666833]
        segments = tuple(
            NarrationSegment(
                segment_id=index,
                text=text,
                instruction="test",
                audio_path=Path(f"actual-{index}.wav"),
                start_seconds=start,
                end_seconds=end,
            )
            for index, (text, start, end) in enumerate(
                zip(texts, starts, ends, strict=True)
            )
        )
        profile = VideoProfile()
        expanded, shots = build_editorial_plan(
            plan=self._plan(),
            package=package,
            segments=segments,
            total_duration=61.666833,
            profile=profile,
        )
        self.assertEqual(len(shots), profile.target_shots)
        self.assertEqual(len(expanded.scenes), profile.target_shots)
        self.assertGreaterEqual(sum(shot.start_seconds < 10.0 for shot in shots), 4)
        self.assertLessEqual(max(shot.duration_seconds for shot in shots), 4.25)
        self.assertEqual(sum(shot.renderer == "wan_i2v" for shot in shots), profile.wan_shots)
        self.assertTrue(any("Infrastructure reuse" in shot.semantic_claim for shot in shots))
        self.assertTrue(
            all(
                shot.semantic_claim in expanded.scenes[shot.shot_id].image_prompt
                for shot in shots
            )
        )

    def test_63_second_canary_uses_twenty_shots_without_overlong_wan(self) -> None:
        texts = [
            "NVIDIA has joined the NSF's State and Regional AI Hubs Program to expand AI research and education.",
            "The initiative provides access to advanced computing, data, and expertise for AI research.",
            "This partnership supports the Genesis Mission and helps expand AI infrastructure across the U.S.",
            "The program will help states and multi-state groups develop AI capabilities.",
            "NVIDIA's involvement is part of a broader effort to advance AI research and education.",
            "The program launches today and focuses on creating a national network for AI innovation.",
            "The initiative ensures that AI research remains accessible to institutions across the country, fostering collaboration and innovation.",
            "The program also emphasizes the importance of education in preparing the next generation of AI professionals.",
            "Before adoption, read the linked source and test the claim on a controlled task.",
            "Track latency, failure rate, human corrections, and repeatability so the decision follows measured behavior rather than a polished announcement.",
        ]
        starts = [
            0.0,
            7.184916666666666,
            12.899583333333332,
            18.862375,
            23.601458333333333,
            29.555083333333336,
            35.493375,
            42.896166666666666,
            49.71058333333333,
            55.666,
        ]
        total_duration = 63.58804166666667
        ends = starts[1:] + [total_duration]
        segments = tuple(
            NarrationSegment(
                segment_id=index,
                text=text,
                instruction="test",
                audio_path=Path(f"failed-canary-{index}.wav"),
                start_seconds=start,
                end_seconds=end,
            )
            for index, (text, start, end) in enumerate(
                zip(texts, starts, ends, strict=True)
            )
        )
        profile = VideoProfile()
        expanded, shots = build_editorial_plan(
            plan=self._plan(),
            package=self._package(),
            segments=segments,
            total_duration=total_duration,
            profile=profile,
        )

        self.assertEqual(len(shots), profile.target_shots)
        self.assertEqual(len(expanded.scenes), profile.target_shots)
        self.assertAlmostEqual(shots[0].duration_seconds, 3.3, places=5)
        self.assertLessEqual(shots[0].duration_seconds, profile.maximum_wan_shot_seconds)
        self.assertEqual(sum(shot.start_seconds < 10.0 for shot in shots), 4)
        self.assertTrue(
            all(
                profile.minimum_shot_seconds <= shot.duration_seconds <= profile.maximum_shot_seconds
                for shot in shots
            )
        )
        wan_shots = [shot for shot in shots if shot.renderer == "wan_i2v"]
        self.assertEqual(len(wan_shots), profile.wan_shots)
        self.assertEqual(wan_shots[0].shot_id, 0)
        self.assertAlmostEqual(sum(shot.duration_seconds for shot in shots), total_duration, places=4)
        self.assertEqual([shot.beat_index for shot in shots], [index for index in range(10) for _ in range(2)])


if __name__ == "__main__":
    unittest.main()
