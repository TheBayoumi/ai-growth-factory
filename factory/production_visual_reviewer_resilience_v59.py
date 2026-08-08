from __future__ import annotations

from typing import Any


_INSTALLED = False


def malformed_reviewer_retry_v59(
    *,
    scene: Any,
    attempt: int,
    error: Exception,
) -> Any:
    """Convert a repeatedly malformed reviewer serialization into a forced scene retry.

    This is deliberately one-way: malformed reviewer output can never approve a frame. The
    generated image is discarded and the normal bounded visual-regeneration loop remains in
    control. If every regenerated image also produces malformed reviewer output, the existing
    attempt ceiling still fails the canary closed.
    """
    from .production_visual_quality import KeyframeReview

    message = str(error)
    text_signal = "text_evidence" in message or "readable" in message or "pseudo-text" in message
    return KeyframeReview(
        scene_index=int(scene.scene_index),
        attempt=int(attempt),
        decision="retry",
        claim_alignment=0.0,
        coherent_scene=False,
        visible_text=text_signal,
        prominent_person=False,
        device_or_panel=False,
        collage_layout=False,
        caption_zone_clear=True,
        reason=(
            "visual reviewer serialization remained malformed after bounded reviewer-only retries; "
            "discard this keyframe and regenerate it"
            + ("; reviewer output also reported readable/pseudo-text evidence" if text_signal else "")
        ),
        repair_instruction=(
            "Regenerate one coherent topic-specific action scene with completely unmarked surfaces and no pseudo-text"
            if text_signal
            else "Regenerate one coherent topic-specific physical action scene with clear subject motion and stable geometry"
        ),
    )


def _is_bounded_malformed_failure(exc: Exception) -> bool:
    message = str(exc)
    return (
        message.startswith("Visual reviewer returned malformed JSON")
        or message.startswith("Visual reviewer could not serialize valid JSON")
    )


def install_production_visual_reviewer_resilience_v59() -> None:
    """Regenerate a scene when Qwen cannot serialize review JSON; never infer approval."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_visual_quality import VisualQualityError
    from .production_visual_semantic_review_v28 import SemanticVisualReviewerV28

    current = SemanticVisualReviewerV28.review
    if getattr(current, "_agf_v59", False):
        _INSTALLED = True
        return

    def review_v59(self: Any, image_path: Any, scene: Any, *, attempt: int, executable_prompt: str) -> Any:
        try:
            return current(
                self,
                image_path,
                scene,
                attempt=attempt,
                executable_prompt=executable_prompt,
            )
        except VisualQualityError as exc:
            if not _is_bounded_malformed_failure(exc):
                raise
            return malformed_reviewer_retry_v59(scene=scene, attempt=attempt, error=exc)

    review_v59._agf_v59 = True  # type: ignore[attr-defined]
    SemanticVisualReviewerV28.review = review_v59
    _INSTALLED = True
