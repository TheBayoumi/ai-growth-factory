from __future__ import annotations

import unittest

from factory.production_visual_quality import VisualQualityError
from factory.production_visual_review_json_v49 import extract_visual_review_json_v49


class VisualReviewJsonV49Tests(unittest.TestCase):
    def test_accepts_fenced_json_object(self) -> None:
        payload = extract_visual_review_json_v49(
            '```json\n{"decision":"approve","claim_alignment":0.91}\n```'
        )
        self.assertEqual(payload["decision"], "approve")
        self.assertEqual(payload["claim_alignment"], 0.91)

    def test_extracts_one_balanced_object_from_surrounding_text(self) -> None:
        payload = extract_visual_review_json_v49(
            'Result follows: {"decision":"retry","visible_text":true} end.'
        )
        self.assertEqual(payload, {"decision": "retry", "visible_text": True})

    def test_repairs_missing_comma_between_object_fields(self) -> None:
        payload = extract_visual_review_json_v49(
            '{\n  "decision": "approve",\n  "claim_alignment": 0.90\n  "semantic_alignment": 0.88,\n  "coherent_scene": true\n}'
        )
        self.assertEqual(payload["claim_alignment"], 0.90)
        self.assertEqual(payload["semantic_alignment"], 0.88)
        self.assertTrue(payload["coherent_scene"])

    def test_repairs_trailing_comma_without_changing_values(self) -> None:
        payload = extract_visual_review_json_v49(
            '{"decision":"retry","reason":"visible pseudo-text",}'
        )
        self.assertEqual(payload["decision"], "retry")
        self.assertEqual(payload["reason"], "visible pseudo-text")

    def test_malformed_value_content_fails_closed_with_typed_error(self) -> None:
        with self.assertRaisesRegex(
            VisualQualityError,
            "Visual reviewer returned malformed JSON after bounded syntax recovery",
        ):
            extract_visual_review_json_v49(
                '{"decision":"retry","reason":"bad "quoted" content","visible_text":true}'
            )

    def test_never_accepts_non_object_json(self) -> None:
        with self.assertRaises(VisualQualityError):
            extract_visual_review_json_v49('[{"decision":"approve"}]')


if __name__ == "__main__":
    unittest.main()
