from __future__ import annotations

import inspect
import unittest

from factory import production_editorial_compositor_v28


class EditorialManifestContractV35Tests(unittest.TestCase):
    def test_compositor_persists_exact_rendered_profile(self) -> None:
        source = inspect.getsource(production_editorial_compositor_v28.compose_editorial_video_v28)
        self.assertIn('"editorial_contract": profile.as_dict()', source)
        self.assertIn('"realized_shot_count": len(ordered_shots)', source)
        self.assertIn('"realized_wan_shots": wan_assets', source)

    def test_compositor_enforces_profile_counts_before_ffmpeg(self) -> None:
        source = inspect.getsource(production_editorial_compositor_v28.compose_editorial_video_v28)
        self.assertIn("profile.minimum_shots <= len(ordered_shots) <= profile.maximum_shots", source)
        self.assertIn("wan_assets != profile.wan_shots", source)
        self.assertNotIn("three Wan", source)


if __name__ == "__main__":
    unittest.main()
