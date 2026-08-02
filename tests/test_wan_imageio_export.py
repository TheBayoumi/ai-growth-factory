import tempfile
import unittest
from pathlib import Path

from PIL import Image

from factory.video_generator import _export_frames


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

    def test_empty_frame_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no frames"):
                _export_frames([], Path(temporary) / "empty.mp4", fps=12)


if __name__ == "__main__":
    unittest.main()
