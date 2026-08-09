from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_remotion_motion_qc_v53 import remotion_motion_failures_v53


def _report(mean: float, near_static: float, jump_ratio: float, maximum: float):
    return SimpleNamespace(
        temporal_window_mean_differences=(mean,),
        temporal_window_near_static_ratios=(near_static,),
        temporal_window_jump_ratios=(jump_ratio,),
        temporal_window_max_differences=(maximum,),
    )


class RemotionMotionQCV53Tests(unittest.TestCase):
    def test_real_canary_smooth_zoom_envelope_passes(self) -> None:
        # Worst reviewed smooth still from canary 20260807T180823Z-91160cd6.
        failures = remotion_motion_failures_v53(
            ("image",),
            _report(3.6180, 0.0, 1.0, 6.1115),
        )
        self.assertEqual((), failures)

    def test_static_image_still_fails(self) -> None:
        failures = remotion_motion_failures_v53(
            ("image",),
            _report(0.05, 0.80, 0.0, 0.07),
        )
        self.assertIn("effectively static", failures[0])

    def test_isolated_pixel_jump_still_fails(self) -> None:
        failures = remotion_motion_failures_v53(
            ("image",),
            _report(0.55, 0.10, 0.18, 4.20),
        )
        self.assertIn("excessive camera motion", failures[0])

    def test_continuously_excessive_motion_still_fails(self) -> None:
        failures = remotion_motion_failures_v53(
            ("image",),
            _report(5.10, 0.0, 1.0, 7.20),
        )
        self.assertIn("excessive camera motion", failures[0])

    def test_wan_thresholds_are_unchanged(self) -> None:
        failures = remotion_motion_failures_v53(
            ("video",),
            _report(0.08, 0.80, 0.0, 5.00),
        )
        self.assertEqual(
            (
                "Wan shot 0 has no meaningful visible motion",
                "Wan shot 0 has unstable frame-to-frame motion",
            ),
            failures,
        )


if __name__ == "__main__":
    unittest.main()
