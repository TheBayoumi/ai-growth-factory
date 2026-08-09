from __future__ import annotations

import unittest

from factory.production_visual_review_transport_v60 import (
    extract_visual_review_payload_v60,
)


class VisualReviewTransportV60Tests(unittest.TestCase):
    def test_live_truncated_nested_text_evidence_forces_retry(self) -> None:
        raw = (
            '{ "semantic_alignment": 0.0, "environment_alignment": 0.0, '
            '"coherent_scene": true, "malformed_subject": false, '
            '"generic_architecture": false, "collage_layout": false, '
            '"text_evidence": [ { "kind": "pseudo", "content": "Calm\\\\r\\\\nPlaye cen'
        )
        payload = extract_visual_review_payload_v60(raw)
        self.assertEqual("retry", payload["decision"])
        self.assertEqual(0.0, payload["claim_alignment"])
        self.assertEqual(0.0, payload["semantic_alignment"])
        self.assertEqual(0.0, payload["setup_alignment"])
        self.assertFalse(payload["coherent_scene"])
        self.assertTrue(payload["visible_text"])
        self.assertTrue(payload["review_transport_recovered"])
        self.assertIn("cannot be approved", payload["reason"])
        self.assertIn("no pseudo-text", payload["repair_instruction"])

    def test_valid_reviewer_json_is_not_modified(self) -> None:
        raw = (
            '{"decision":"approve","claim_alignment":0.92,'
            '"semantic_alignment":0.88,"setup_alignment":0.84,'
            '"coherent_scene":true,"visible_text":false,'
            '"malformed_subject":false,"generic_architecture":false,'
            '"collage_layout":false,"reason":"","repair_instruction":""}'
        )
        payload = extract_visual_review_payload_v60(raw)
        self.assertEqual("approve", payload["decision"])
        self.assertEqual(0.92, payload["claim_alignment"])
        self.assertNotIn("review_transport_recovered", payload)

    def test_valid_non_object_payload_remains_fail_closed_as_retry(self) -> None:
        payload = extract_visual_review_payload_v60('[{"decision":"approve"}]')
        self.assertEqual("retry", payload["decision"])
        self.assertEqual(0.0, payload["claim_alignment"])
        self.assertTrue(payload["review_transport_recovered"])


if __name__ == "__main__":
    unittest.main()
