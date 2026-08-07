from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from factory.models import (
    AudioReview,
    FailedSegment,
    NarrationSegment,
    ReviewScores,
)
from factory.production_voice_review_convergence_v47 import (
    build_voice_review_convergence_class_v47,
    cadence_only_failure_v47,
    threshold_failure_for_result_v47,
)


def _scores(**overrides: float) -> dict[str, float]:
    values = {
        "script_fidelity": 1.0,
        "naturalness": 1.0,
        "authority": 1.0,
        "engagement": 1.0,
        "pronunciation": 1.0,
        "pace": 1.0,
        "pause_quality": 1.0,
        "emotional_match": 1.0,
        "audio_artifacts": 1.0,
    }
    values.update(overrides)
    return values


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        reviewer_overall_threshold=0.90,
        reviewer_fidelity_threshold=0.98,
        reviewer_naturalness_threshold=0.85,
        reviewer_pronunciation_threshold=0.92,
    )


def _review(
    *,
    decision: str,
    failures: tuple[FailedSegment, ...] = (),
    raw: list[dict[str, Any]] | None = None,
    fidelity: float = 1.0,
) -> AudioReview:
    return AudioReview(
        decision=decision,
        overall_score=0.95,
        scores=ReviewScores.from_dict(_scores(script_fidelity=fidelity)),
        failed_segments=failures,
        summary="review",
        reviewer_model="fake",
        raw_response=json.dumps(raw or []),
    )


class ProductionVoiceReviewConvergenceV47Tests(unittest.TestCase):
    def test_threshold_failure_targets_only_underperforming_segment(self) -> None:
        failure = threshold_failure_for_result_v47(
            {
                "segment_id": 4,
                "decision": "approve",
                "overall_score": 0.95,
                "scores": _scores(script_fidelity=0.96),
            },
            _settings(),
        )

        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.segment_id, 4)
        self.assertIn("script fidelity", failure.reason)
        self.assertIn("every supplied word", failure.tts_instruction)

        healthy = threshold_failure_for_result_v47(
            {
                "segment_id": 5,
                "decision": "approve",
                "overall_score": 0.95,
                "scores": _scores(),
            },
            _settings(),
        )
        self.assertIsNone(healthy)

    def test_wrapper_converts_aggregate_miss_to_segment_local_retry(self) -> None:
        raw = [
            {
                "segment_id": 0,
                "decision": "approve",
                "overall_score": 0.95,
                "scores": _scores(script_fidelity=0.95),
            },
            {
                "segment_id": 1,
                "decision": "approve",
                "overall_score": 0.97,
                "scores": _scores(),
            },
        ]

        class BaseReviewer:
            def __init__(self, settings: Any) -> None:
                self.settings = settings

            def review(self, **_kwargs: Any) -> AudioReview:
                return _review(
                    decision="approve",
                    raw=raw,
                    fidelity=0.975,
                )

        Reviewer = build_voice_review_convergence_class_v47(BaseReviewer)
        reviewer = Reviewer(_settings())
        segments = [
            NarrationSegment(0, "first segment", "", Path("first.wav"), attempt=2),
            NarrationSegment(1, "second segment", "", Path("second.wav"), attempt=2),
        ]

        result = reviewer.review(segments=segments)

        self.assertEqual(result.decision, "retry_segments")
        self.assertEqual(
            tuple(failure.segment_id for failure in result.failed_segments),
            (0,),
        )

    def test_unchanged_approved_generation_ignores_later_cadence_flip(self) -> None:
        cadence_failure = FailedSegment(
            segment_id=5,
            reason="Naturalness and pace are slightly off because the cadence feels rushed.",
            tts_instruction=(
                "Preserve the overall pace and improve only clause-boundary pauses "
                "and natural phrasing."
            ),
        )

        class BaseReviewer:
            def __init__(self, settings: Any) -> None:
                self.settings = settings
                self.calls = 0

            def review(self, **_kwargs: Any) -> AudioReview:
                self.calls += 1
                if self.calls == 1:
                    return _review(
                        decision="approve",
                        raw=[
                            {
                                "segment_id": 5,
                                "decision": "approve",
                                "overall_score": 0.97,
                                "scores": _scores(),
                            }
                        ],
                    )
                return _review(
                    decision="retry_segments",
                    failures=(cadence_failure,),
                    raw=[
                        {
                            "segment_id": 5,
                            "decision": "retry",
                            "overall_score": 0.94,
                            "scores": _scores(),
                        }
                    ],
                )

        Reviewer = build_voice_review_convergence_class_v47(BaseReviewer)
        reviewer = Reviewer(_settings())
        segment = NarrationSegment(
            5,
            "The expansion is part of the strategy,",
            "",
            Path("segment.wav"),
            attempt=2,
        )

        first = reviewer.review(segments=[segment])
        second = reviewer.review(segments=[segment])

        self.assertEqual(first.decision, "approve")
        self.assertEqual(second.decision, "approve")
        self.assertEqual(second.failed_segments, ())
        self.assertIn("unchanged previously approved", second.summary)

    def test_critical_failure_is_never_cleared(self) -> None:
        self.assertFalse(
            cadence_only_failure_v47(
                FailedSegment(
                    segment_id=5,
                    reason="A word is missing and the cadence is rushed.",
                    tts_instruction="Restore the omitted word and improve the pause.",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
