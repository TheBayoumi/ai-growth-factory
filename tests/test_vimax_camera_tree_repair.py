from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_vimax_planner import sanitize_camera_parent_items


class ViMaxCameraTreeRepairTests(unittest.TestCase):
    def _cameras(self):
        return [
            SimpleNamespace(idx=0, active_shot_idxs=[0, 1]),
            SimpleNamespace(idx=1, active_shot_idxs=[2, 3]),
            SimpleNamespace(idx=2, active_shot_idxs=[4, 5]),
            SimpleNamespace(idx=3, active_shot_idxs=[6, 7]),
        ]

    @staticmethod
    def _item(parent, shot, reason="valid"):
        return SimpleNamespace(
            parent_cam_idx=parent,
            parent_shot_idx=shot,
            reason=reason,
            is_parent_fully_covers_child=True,
            missing_info=None,
        )

    def test_repairs_root_self_parent_and_extra_root(self) -> None:
        normalized, repairs = sanitize_camera_parent_items(
            self._cameras(),
            [
                self._item(0, 0),
                None,
                self._item(1, 3),
                self._item(2, 5),
            ],
        )
        self.assertIsNone(normalized[0])
        self.assertEqual(0, normalized[1]["parent_cam_idx"])
        self.assertEqual(1, normalized[1]["parent_shot_idx"])
        self.assertIn("root_camera_had_parent", {item["issue"] for item in repairs})
        self.assertIn("extra_root", {item["issue"] for item in repairs})

    def test_preserves_valid_edges(self) -> None:
        normalized, repairs = sanitize_camera_parent_items(
            self._cameras(),
            [
                None,
                self._item(0, 1),
                self._item(1, 3),
                self._item(2, 5),
            ],
        )
        self.assertEqual([], repairs)
        self.assertEqual([None, 0, 1, 2], [
            None if item is None else item["parent_cam_idx"] for item in normalized
        ])
        self.assertEqual([None, 1, 3, 5], [
            None if item is None else item["parent_shot_idx"] for item in normalized
        ])

    def test_repairs_invalid_parent_shot(self) -> None:
        normalized, repairs = sanitize_camera_parent_items(
            self._cameras(),
            [
                None,
                self._item(0, 999),
                self._item(1, 3),
                self._item(2, 5),
            ],
        )
        self.assertEqual(1, normalized[1]["parent_shot_idx"])
        self.assertIn("invalid_parent_shot", {item["issue"] for item in repairs})

    def test_breaks_cycle_without_changing_other_edges(self) -> None:
        normalized, repairs = sanitize_camera_parent_items(
            self._cameras(),
            [
                None,
                self._item(2, 4),
                self._item(1, 2),
                self._item(2, 5),
            ],
        )
        parents = [None if item is None else item["parent_cam_idx"] for item in normalized]
        self.assertEqual([None, 2, 0, 2], parents)
        self.assertIn("cycle", {item["issue"] for item in repairs})

    def test_rejects_response_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            sanitize_camera_parent_items(self._cameras(), [None])


if __name__ == "__main__":
    unittest.main()
