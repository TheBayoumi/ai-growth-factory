import unittest

from factory.production_reviewer_feedback import normalize_retry_feedback


class ProductionReviewerFeedbackTests(unittest.TestCase):
    def test_missing_retry_fields_are_derived_from_lowest_score(self):
        result = normalize_retry_feedback(
            {
                "decision": "retry",
                "overall_score": 0.79,
                "scores": {
                    "script_fidelity": 0.99,
                    "naturalness": 0.61,
                    "authority": 0.90,
                    "engagement": 0.86,
                    "pronunciation": 0.94,
                    "pace": 0.88,
                    "pause_quality": 0.91,
                    "emotional_match": 0.84,
                    "audio_artifacts": 0.96,
                },
                "reason": "",
                "tts_instruction": "",
            }
        )

        self.assertIn("naturalness", result["reason"])
        self.assertIn("natural human phrasing", result["tts_instruction"])
        self.assertEqual(result["decision"], "retry")
        self.assertEqual(result["overall_score"], 0.79)

    def test_schema_placeholder_is_replaced_by_scored_instruction(self):
        result = normalize_retry_feedback(
            {
                "decision": "retry",
                "scores": {
                    "script_fidelity": 0.98,
                    "naturalness": 0.54,
                    "authority": 0.90,
                    "engagement": 0.62,
                    "pronunciation": 0.91,
                    "pace": 0.88,
                    "pause_quality": 0.90,
                    "emotional_match": 0.85,
                    "audio_artifacts": 0.95,
                },
                "reason": "The delivery needs more naturalness.",
                "tts_instruction": "standalone Qwen3-TTS repair instruction, or empty when approved",
            }
        )
        self.assertEqual(result["reason"], "The delivery needs more naturalness.")
        self.assertIn("natural human phrasing", result["tts_instruction"])
        self.assertNotIn("standalone", result["tts_instruction"])

    def test_existing_actionable_feedback_is_preserved(self):
        original = {
            "decision": "retry",
            "scores": {"pace": 0.70},
            "reason": "The middle clause is rushed.",
            "tts_instruction": "Slow the middle clause without changing any words.",
        }
        self.assertEqual(normalize_retry_feedback(original), original)

    def test_non_retry_decision_is_not_modified(self):
        original = {
            "decision": "approve",
            "scores": {},
            "reason": "",
            "tts_instruction": "",
        }
        self.assertIs(normalize_retry_feedback(original), original)

    def test_invalid_score_values_fall_back_safely(self):
        result = normalize_retry_feedback(
            {
                "decision": "retry",
                "scores": {"script_fidelity": "not-a-number"},
            }
        )
        self.assertTrue(result["reason"])
        self.assertTrue(result["tts_instruction"])


if __name__ == "__main__":
    unittest.main()
