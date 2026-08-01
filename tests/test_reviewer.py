import unittest
from unittest.mock import patch

from factory.config import Settings
from factory.models import AudioReview
from factory.reviewer import _extract_json, review_passes


class ReviewerTests(unittest.TestCase):
    def test_extracts_json_from_fenced_output(self):
        data = _extract_json('```json\n{"decision":"approve"}\n```')
        self.assertEqual(data["decision"], "approve")

    def test_thresholds_are_enforced_locally(self):
        payload = {
            "decision": "approve",
            "overall_score": 0.90,
            "scores": {
                "script_fidelity": 0.99,
                "naturalness": 0.90,
                "authority": 0.90,
                "engagement": 0.90,
                "pronunciation": 0.95,
                "pace": 0.90,
                "pause_quality": 0.90,
                "emotional_match": 0.90,
                "audio_artifacts": 0.95,
            },
            "failed_segments": [],
            "summary": "Publication quality.",
        }
        review = AudioReview.from_dict(payload, model="gpt-realtime-2.1")
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(review_passes(review, Settings.from_env()))
        payload["scores"]["script_fidelity"] = 0.90
        weak = AudioReview.from_dict(payload, model="gpt-realtime-2.1")
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(review_passes(weak, Settings.from_env()))


if __name__ == "__main__":
    unittest.main()
