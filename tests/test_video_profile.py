from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from factory.video_profile import VideoProfile


class VideoProfileTests(unittest.TestCase):
    def test_defaults_are_fail_closed(self) -> None:
        profile = VideoProfile()
        profile.validate()
        self.assertEqual(profile.target_wpm, 142)
        self.assertFalse(profile.allow_asset_looping)
        self.assertFalse(profile.allow_destructive_caption_matte)
        self.assertGreaterEqual(profile.minimum_shots, 14)
        self.assertLessEqual(profile.maximum_tempo_factor, 1.15)

    def test_json_and_scalar_overrides_are_validated(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VIDEO_PROFILE_JSON": '{"target_shots": 17, "maximum_shots": 19}',
                "VIDEO_TARGET_WPM": "140",
                "VIDEO_MIN_WPM": "136",
                "VIDEO_MAX_WPM": "144",
            },
            clear=False,
        ):
            profile = VideoProfile.from_env()
        self.assertEqual(profile.target_shots, 17)
        self.assertEqual(profile.target_wpm, 140)

    def test_looping_cannot_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "never permits source-asset looping"):
            VideoProfile(allow_asset_looping=True).validate()

    def test_destructive_matte_cannot_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "never permits destructive caption mattes"):
            VideoProfile(allow_destructive_caption_matte=True).validate()


if __name__ == "__main__":
    unittest.main()
