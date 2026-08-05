from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

from factory.editorial_timeline import StoryBeat, _shot_counts
from factory.production_editorial_wan_allocator_v38 import (
    allocate_editorial_durations_v38,
)
from factory.video_profile import VideoProfile


class ProductionEditorialWanAllocatorV38Tests(unittest.TestCase):
    def _failed_canary_beats(self) -> list[StoryBeat]:
        starts = [
            0.0,
            7.660125,
            13.220125,
            20.421,
            26.804458333333333,
            32.33779166666667,
            37.861958333333334,
            44.256625,
            49.827875,
            54.955958333333335,
        ]
        total_duration = 60.082625
        ends = starts[1:] + [total_duration]
        return [
            StoryBeat(
                beat_id=index,
                segment_id=index,
                sentence_index=0,
                start_seconds=start,
                duration_seconds=end - start,
                narration_text=f"Failed canary narration beat {index}",
                scene_candidates=(0,),
            )
            for index, (start, end) in enumerate(zip(starts, ends, strict=True))
        ]

    def test_exact_766_second_opening_uses_spare_capacity_without_relaxing_bounds(self) -> None:
        profile = VideoProfile()
        beats = self._failed_canary_beats()
        counts = _shot_counts(beats, profile)
        self.assertEqual(counts, [2] * 10)

        allocated = allocate_editorial_durations_v38(beats, counts, profile)

        self.assertEqual(counts[0], 3)
        self.assertEqual(sum(counts), 21)
        self.assertLessEqual(allocated[0][0], profile.maximum_wan_shot_seconds)
        self.assertTrue(
            all(
                profile.minimum_shot_seconds - 1e-6
                <= duration
                <= profile.maximum_shot_seconds + 1e-6
                for durations in allocated
                for duration in durations
            )
        )
        for beat, durations in zip(beats, allocated, strict=True):
            self.assertAlmostEqual(sum(durations), beat.duration_seconds, places=6)

        starts: list[float] = []
        for beat, durations in zip(beats, allocated, strict=True):
            cursor = beat.start_seconds
            for duration in durations:
                starts.append(cursor)
                cursor += duration
        self.assertGreaterEqual(
            sum(start < 10.0 for start in starts),
            profile.first_ten_seconds_minimum_shots,
        )

    def test_allocator_remains_fail_closed_when_hard_capacity_is_exhausted(self) -> None:
        profile = replace(VideoProfile(), maximum_shots=20)
        beats = self._failed_canary_beats()
        counts = _shot_counts(beats, profile)
        with self.assertRaisesRegex(ValueError, "Opening Wan shot cannot fit"):
            allocate_editorial_durations_v38(beats, counts, profile)

    def test_runtime_installs_v38_before_render_execution(self) -> None:
        from factory import production_runtime

        source = inspect.getsource(production_runtime.install_production_runtime)
        self.assertIn("install_production_editorial_wan_allocator_v38", source)
        self.assertLess(
            source.index("install_production_editorial_wan_allocator_v38()"),
            source.index("install_production_visual_runtime_v28()"),
        )


if __name__ == "__main__":
    unittest.main()
