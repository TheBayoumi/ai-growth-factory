from __future__ import annotations

import unittest

from factory.production_caption_scale_v67 import (
    fit_caption_lines_v67,
    proportional_caption_margin_v67,
)


class ProductionCaptionScaleV67Tests(unittest.TestCase):
    def test_preflight_720p_uses_proportional_not_legacy_84px_margin(self) -> None:
        self.assertEqual(65, proportional_caption_margin_v67(720))
        layout = fit_caption_lines_v67(
            "This move strengthens the region's",
            width=720,
            height=1280,
        )
        self.assertLessEqual(layout["maximum_line_width_pixels"], layout["safe_width_pixels"])
        self.assertEqual(65, layout["horizontal_margin_pixels"])

    def test_release_1080p_geometry_is_unchanged(self) -> None:
        self.assertEqual(97, proportional_caption_margin_v67(1080))
        layout = fit_caption_lines_v67(
            "This move strengthens the region's",
            width=1080,
            height=1920,
        )
        self.assertLessEqual(layout["maximum_line_width_pixels"], layout["safe_width_pixels"])
        self.assertEqual(97, layout["horizontal_margin_pixels"])

    def test_invalid_width_fails(self) -> None:
        with self.assertRaises(ValueError):
            proportional_caption_margin_v67(0)


if __name__ == "__main__":
    unittest.main()
