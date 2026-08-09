from __future__ import annotations

import unittest
from types import SimpleNamespace

from factory.production_visual_reviewer_resilience_v59 import (
    malformed_reviewer_retry_v59,
)


class VisualReviewerResilienceV59Tests(unittest.TestCase):
    def test_malformed_reviewer_output_can_only_force_retry(self) -> None:
        review = malformed_reviewer_retry_v59(
            scene=SimpleNamespace(scene_index=7),
            attempt=3,
            error=RuntimeError(
                'Visual reviewer could not serialize valid JSON; excerpt={"semantic_alignment": 0.0, "text_evidence": [{"kind": "readable"}]'
            ),
        )
        self.assertEqual("retry", review.decision)
        self.assertEqual(0.0, review.claim_alignment)
        self.assertFalse(review.coherent_scene)
        self.assertTrue(review.visible_text)
        self.assertIn("discard this keyframe", review.reason)
        self.assertIn("no pseudo-text", review.repair_instruction)

    def test_malformed_without_text_signal_still_forces_retry(self) -> None:
        review = malformed_reviewer_retry_v59(
            scene=SimpleNamespace(scene_index=2),
            attempt=1,
            error=RuntimeError("Visual reviewer returned malformed JSON after bounded syntax recovery"),
        )
        self.assertEqual("retry", review.decision)
        self.assertFalse(review.visible_text)
        self.assertIn("stable geometry", review.repair_instruction)


if __name__ == "__main__":
    unittest.main()
