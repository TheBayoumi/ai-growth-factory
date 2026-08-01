from __future__ import annotations

import unittest

from factory.models import AudioReview, ReviewScores, VoiceContract


class ModelTests(unittest.TestCase):
    def test_voice_contract_validates(self) -> None:
        VoiceContract().validate()
        with self.assertRaises(ValueError):
            VoiceContract(target_wpm=300).validate()

    def test_review_parses(self) -> None:
        scores = {name: 0.95 for name in ReviewScores.__dataclass_fields__}
        review = AudioReview.from_dict({"decision": "approve", "overall_score": 0.95, "scores": scores, "failed_segments": [], "summary": "ok"}, "test")
        self.assertEqual(review.decision, "approve")


if __name__ == "__main__":
    unittest.main()
