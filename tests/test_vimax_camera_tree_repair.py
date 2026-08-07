from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_vimax_planner import (
    normalize_camera_tree_response_payload,
    parse_camera_tree_response,
    sanitize_camera_parent_items,
)


class _FakeCameraTreeResponse:
    @classmethod
    def model_validate(cls, payload):
        return SimpleNamespace(camera_parent_items=payload["camera_parent_items"])


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

    def test_normalizes_integer_parent_shorthand_before_pydantic(self) -> None:
        response, repairs = parse_camera_tree_response(
            SimpleNamespace(
                content=(
                    "```json\n"
                    '{"camera_parent_items":[0,0,1,1]}'
                    "\n```"
                )
            ),
            response_model=_FakeCameraTreeResponse,
        )

        self.assertEqual(
            [0, 0, 1, 1],
            [item["parent_cam_idx"] for item in response.camera_parent_items],
        )
        self.assertTrue(
            all(item["parent_shot_idx"] is None for item in response.camera_parent_items)
        )
        self.assertEqual(
            ["integer_parent_shorthand"] * 4,
            [item["issue"] for item in repairs],
        )

        normalized, tree_repairs = sanitize_camera_parent_items(
            self._cameras(),
            response.camera_parent_items,
        )
        self.assertIsNone(normalized[0])
        self.assertEqual([None, 0, 1, 1], [
            None if item is None else item["parent_cam_idx"] for item in normalized
        ])
        self.assertIn(
            "invalid_parent_shot",
            {item["issue"] for item in tree_repairs},
        )

    def test_preserves_full_parent_metadata_objects(self) -> None:
        payload, repairs = normalize_camera_tree_response_payload(
            {
                "camera_parent_items": [
                    None,
                    {
                        "parent_cam_idx": 0,
                        "parent_shot_idx": 1,
                        "reason": "wide shot covers close-up",
                        "is_parent_fully_covers_child": True,
                        "missing_info": None,
                    },
                ]
            }
        )
        self.assertEqual([], repairs)
        self.assertIsNone(payload["camera_parent_items"][0])
        self.assertEqual(
            "wide shot covers close-up",
            payload["camera_parent_items"][1]["reason"],
        )

    def test_rejects_boolean_parent_shorthand(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean"):
            normalize_camera_tree_response_payload(
                {"camera_parent_items": [None, True]}
            )


if __name__ == "__main__":
    unittest.main()
