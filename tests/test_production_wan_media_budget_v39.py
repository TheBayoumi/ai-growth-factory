from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.video_generator import VideoGenerationError, _validate_wan_media_budget


class ProductionWanMediaBudgetV39Tests(unittest.TestCase):
    @staticmethod
    def _plan(*modes: str) -> SimpleNamespace:
        return SimpleNamespace(
            scenes=tuple(SimpleNamespace(generation_mode=mode) for mode in modes)
        )

    @staticmethod
    def _assets(*media_types: str) -> tuple[SimpleNamespace, ...]:
        return tuple(SimpleNamespace(media_type=media_type) for media_type in media_types)

    def test_six_clip_profile_budget_is_accepted(self) -> None:
        plan = self._plan(*(["wan_i2v"] * 6), *(["image"] * 15))
        assets = self._assets(*(["video"] * 6), *(["image"] * 15))

        self.assertEqual(_validate_wan_media_budget(plan, assets), (6, 6))

    def test_realized_budget_must_match_plan(self) -> None:
        plan = self._plan("wan_i2v", "wan_i2v", "image")
        assets = self._assets("video", "image", "image")

        with self.assertRaisesRegex(
            VideoGenerationError,
            "requires 2 Wan clips; generated 1",
        ):
            _validate_wan_media_budget(plan, assets)

    def test_plan_without_wan_clip_fails_closed(self) -> None:
        with self.assertRaisesRegex(VideoGenerationError, "at least one Wan clip"):
            _validate_wan_media_budget(
                self._plan("image", "image"),
                self._assets("image", "image"),
            )


if __name__ == "__main__":
    unittest.main()
