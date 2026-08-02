import unittest
from unittest.mock import patch

from factory import production_video_qc


class ProductionVideoQCV10Tests(unittest.TestCase):
    @staticmethod
    def _hold_jump_frames() -> list[bytes]:
        frame_size = 90 * round(160 * 0.68)
        quiet = bytes([0]) * frame_size
        jump = bytes([255]) * frame_size
        return [quiet, quiet, quiet, jump, quiet, quiet, quiet, jump]

    def test_caption_region_is_removed_before_motion_analysis(self):
        full_frame = bytes(range(256)) * ((90 * 160 // 256) + 1)
        full_frame = full_frame[: 90 * 160]
        with patch.object(
            production_video_qc,
            "_ORIGINAL_RAW_GRAY_FRAMES",
            return_value=[full_frame],
        ):
            result = production_video_qc._upper_visual_frames(
                "video.mp4", center=1.0
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 90 * round(160 * 0.68))
        self.assertEqual(result[0], full_frame[: len(result[0])])

    def test_intentional_image_scene_is_not_classified_as_video_stutter(self):
        token = production_video_qc._MEDIA_TYPES.set(("image",))
        try:
            with patch.object(
                production_video_qc.video_qc,
                "_scene_centers",
                return_value=[1.0],
            ), patch.object(
                production_video_qc.video_qc,
                "_raw_gray_frames",
                return_value=self._hold_jump_frames(),
            ):
                report = production_video_qc._production_temporal_stability(
                    "video.mp4", 2.0, [2.0]
                )
        finally:
            production_video_qc._MEDIA_TYPES.reset(token)
        self.assertEqual(report[-1], 0)
        self.assertGreater(report[2][0], 0.0)

    def test_same_hold_jump_pattern_still_fails_for_wan_video_scene(self):
        token = production_video_qc._MEDIA_TYPES.set(("video",))
        try:
            with patch.object(
                production_video_qc.video_qc,
                "_scene_centers",
                return_value=[1.0],
            ), patch.object(
                production_video_qc.video_qc,
                "_raw_gray_frames",
                return_value=self._hold_jump_frames(),
            ):
                report = production_video_qc._production_temporal_stability(
                    "video.mp4", 2.0, [2.0]
                )
        finally:
            production_video_qc._MEDIA_TYPES.reset(token)
        self.assertEqual(report[-1], 1)

    def test_invalid_media_type_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported scene media type"):
            production_video_qc.verify_production_video_output(
                object(),
                "video.mp4",
                "thumbnail.png",
                scene_media_types=["unknown"],
            )


if __name__ == "__main__":
    unittest.main()
