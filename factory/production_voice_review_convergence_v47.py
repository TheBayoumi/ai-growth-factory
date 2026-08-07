from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from .models import AudioReview, FailedSegment, NarrationSegment, ReviewScores


_INSTALLED = False
_PACE_OR_CADENCE_RE = re.compile(
    r"\b(?:pace|faster|slower|speed|wpm|words per minute|pause|pauses|"
    r"cadence|phrasing|naturalness|rhythm|rushed|dragging)\b",
    re.IGNORECASE,
)
_CRITICAL_RE = re.compile(
    r"\b(?:omit|omission|missing|insert|addition|repeat|substitut|paraphras|"
    r"mispronoun|pronunciation|noise|click|distort|artifact|metallic|corrupt|"
    r"garbl|clipp|word mismatch|script fidelity)\w*\b",
    re.IGNORECASE,
)


def _score_issue(
    label: str,
    value: float,
    threshold: float,
) -> str:
    return f"{label} {value:.3f} is below the required {threshold:.3f}"


def threshold_failure_for_result_v47(
    result: dict[str, Any],
    settings: Any,
) -> FailedSegment | None:
    """Map one locally under-threshold segment to one executable repair."""
    if str(result.get("decision") or "").strip().casefold() != "approve":
        return None

    segment_id = int(result["segment_id"])
    overall = float(result.get("overall_score") or 0.0)
    scores = ReviewScores.from_dict(dict(result.get("scores") or {}))
    reasons: list[str] = []
    instructions: list[str] = []

    if overall < float(settings.reviewer_overall_threshold):
        reasons.append(
            _score_issue(
                "overall score",
                overall,
                float(settings.reviewer_overall_threshold),
            )
        )
        instructions.append(
            "Improve publication quality while preserving the exact transcript and voice contract"
        )
    if scores.script_fidelity < float(settings.reviewer_fidelity_threshold):
        reasons.append(
            _score_issue(
                "script fidelity",
                scores.script_fidelity,
                float(settings.reviewer_fidelity_threshold),
            )
        )
        instructions.append(
            "Speak every supplied word exactly once with no omissions, additions, substitutions, or repetitions"
        )
    if scores.naturalness < float(settings.reviewer_naturalness_threshold):
        reasons.append(
            _score_issue(
                "naturalness",
                scores.naturalness,
                float(settings.reviewer_naturalness_threshold),
            )
        )
        instructions.append(
            "Use smooth human phrasing and natural clause-boundary pauses without changing the overall pace"
        )
    if scores.pronunciation < float(settings.reviewer_pronunciation_threshold):
        reasons.append(
            _score_issue(
                "pronunciation",
                scores.pronunciation,
                float(settings.reviewer_pronunciation_threshold),
            )
        )
        instructions.append(
            "Articulate names, numbers, acronyms, and technical terms clearly"
        )
    if scores.audio_artifacts < 0.90:
        reasons.append(
            _score_issue("audio artifacts", scores.audio_artifacts, 0.90)
        )
        instructions.append(
            "Produce clean boundaries without clicks, distortion, metallic resonance, or synthetic strain"
        )

    if not reasons:
        return None
    return FailedSegment(
        segment_id=segment_id,
        reason="; ".join(reasons) + ".",
        tts_instruction=". ".join(dict.fromkeys(instructions)) + ".",
    )


def _raw_segment_results(review: AudioReview) -> tuple[dict[str, Any], ...]:
    if not review.raw_response.strip():
        return ()
    try:
        value = json.loads(review.raw_response)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def cadence_only_failure_v47(failure: FailedSegment) -> bool:
    """Return true only for subjective timing/naturalness feedback."""
    combined = f"{failure.reason} {failure.tts_instruction}"
    return bool(_PACE_OR_CADENCE_RE.search(combined)) and not bool(
        _CRITICAL_RE.search(combined)
    )


def build_voice_review_convergence_class_v47(base_reviewer: type) -> type:
    """Keep review retries segment-local and stable across unchanged generations."""

    class StableSegmentReviewerV47(base_reviewer):
        def __init__(self, settings: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(settings, *args, **kwargs)
            self._approved_generation_attempts_v47: dict[int, int] = {}

        def review(self, **kwargs: Any) -> AudioReview:
            review = super().review(**kwargs)
            if review.decision == "reject":
                return review

            segment_list = tuple(kwargs.get("segments") or ())
            segments = {
                int(segment.segment_id): segment
                for segment in segment_list
                if isinstance(segment, NarrationSegment)
            }
            failures = {
                int(failure.segment_id): failure
                for failure in review.failed_segments
            }

            # Qwen returns per-segment scores in raw_response. Enforce local thresholds
            # on those exact segments instead of regenerating every segment because one
            # aggregate score missed a threshold.
            for result in _raw_segment_results(review):
                try:
                    segment_id = int(result["segment_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if segment_id in failures or segment_id not in segments:
                    continue
                threshold_failure = threshold_failure_for_result_v47(
                    result,
                    self.settings,
                )
                if threshold_failure is not None:
                    failures[segment_id] = threshold_failure

            cleared: list[int] = []
            stable_failures: dict[int, FailedSegment] = {}
            for segment_id, failure in failures.items():
                segment = segments.get(segment_id)
                previously_approved_attempt = (
                    self._approved_generation_attempts_v47.get(segment_id)
                )
                if (
                    segment is not None
                    and previously_approved_attempt == int(segment.attempt)
                    and cadence_only_failure_v47(failure)
                ):
                    cleared.append(segment_id)
                    continue
                stable_failures[segment_id] = failure

            for segment_id, segment in segments.items():
                if segment_id not in stable_failures:
                    self._approved_generation_attempts_v47[segment_id] = int(
                        segment.attempt
                    )

            ordered_failures = tuple(
                stable_failures[segment_id]
                for segment_id in sorted(stable_failures)
            )
            decision = "retry_segments" if ordered_failures else "approve"
            summary = review.summary
            if cleared:
                cleared_text = ", ".join(str(item) for item in sorted(cleared))
                summary = (
                    f"{summary} Cadence-only reviewer flips were cleared for "
                    f"unchanged previously approved generation(s): {cleared_text}."
                ).strip()
            elif ordered_failures and not review.failed_segments:
                summary = (
                    "Local per-segment score thresholds identified "
                    f"{len(ordered_failures)} targeted repair(s)."
                )

            return replace(
                review,
                decision=decision,
                failed_segments=ordered_failures,
                summary=summary,
            )

    StableSegmentReviewerV47.__name__ = "StableSegmentReviewerV47"
    return StableSegmentReviewerV47


def install_production_voice_review_convergence_v47() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import voice_pipeline

    voice_pipeline.QwenOmniReviewer = build_voice_review_convergence_class_v47(
        voice_pipeline.QwenOmniReviewer
    )
    _INSTALLED = True
