import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from factory.video_generator import _export_frames, _frame_to_uint8


class WanImageIOExportTests(unittest.TestCase):
    def test_frames_export_to_playable_mp4(self):
        frames = [
            Image.new("RGB", (64, 64), (index * 40, 80, 160 - index * 20))
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wan.mp4"
            result = _export_frames(frames, output, fps=12)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 500)

    def test_float_hwc_frames_are_scaled_to_rgb_uint8_and_exported(self):
        frames = [
            np.full((48, 64, 3), fill_value=index / 3.0, dtype=np.float32)
            for index in range(4)
        ]
        normalized = _frame_to_uint8(frames[-1])
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, (48, 64, 3))
        self.assertEqual(int(normalized.max()), 255)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "float-wan.mp4"
            _export_frames(frames, output, fps=12)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 500)

    def test_channel_first_negative_range_frame_is_normalized(self):
        frame = np.zeros((3, 12, 16), dtype=np.float32)
        frame[0] = -1.0
        frame[1] = 0.0
        frame[2] = 1.0

        normalized = _frame_to_uint8(frame)

        self.assertEqual(normalized.shape, (12, 16, 3))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(int(normalized[0, 0, 0]), 0)
        self.assertIn(int(normalized[0, 0, 1]), {127, 128})
        self.assertEqual(int(normalized[0, 0, 2]), 255)

    def test_inconsistent_frame_dimensions_fail_closed(self):
        frames = [
            np.zeros((16, 16, 3), dtype=np.float32),
            np.zeros((18, 16, 3), dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "inconsistent dimensions"):
                _export_frames(frames, Path(temporary) / "bad.mp4", fps=12)

    def test_empty_frame_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no frames"):
                _export_frames([], Path(temporary) / "empty.mp4", fps=12)


if __name__ == "__main__":
    unittest.main()
