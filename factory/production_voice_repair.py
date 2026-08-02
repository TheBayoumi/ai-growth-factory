from __future__ import annotations

from dataclasses import replace

from .models import AudioReview


_INSTALLED = False


def promote_repairable_reject(review: AudioReview) -> AudioReview:
    """Convert a recoverable segment rejection into the existing repair path.

    The reviewer may label a track ``reject`` when one segment omits or changes words,
    while still returning an exact segment ID and a concrete TTS repair instruction.
    That feedback is actionable. Promoting only those reviews to ``retry_segments``
    lets the bounded voice loop regenerate the named segment. Reviews without complete
    repair instructions remain hard rejects, and the final attempt still fails closed.
    """
    if review.decision != "reject" or not review.failed_segments:
        return review
    if any(
        not failure.reason.strip() or not failure.tts_instruction.strip()
        for failure in review.failed_segments
    ):
        return review
    return replace(review, decision="retry_segments")


def install_production_voice_repair() -> None:
    """Install bounded repair handling for the production Qwen Omni reviewer."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .qwen_omni_reviewer import QwenOmniReviewer

    original_review = QwenOmniReviewer.review

    def production_review(self, *args, **kwargs):
        return promote_repairable_reject(original_review(self, *args, **kwargs))

    QwenOmniReviewer.review = production_review
    _INSTALLED = True
