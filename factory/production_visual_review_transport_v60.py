from __future__ import annotations

import hashlib
from typing import Any


_INSTALLED = False


def _transport_retry_payload(text: str) -> dict[str, Any]:
    """Fail closed at the reviewer transport boundary without aborting the scene loop.

    A malformed or truncated reviewer response is never evidence of quality.  Return a complete
    retry payload with zero alignment so the existing bounded keyframe-regeneration loop can
    continue.  This deliberately does not attempt to infer an approval from partial model text.
    """
    clean = str(text or "").strip()
    digest = hashlib.sha256(clean.encode("utf-8", errors="replace")).hexdigest()
    lowered = clean.casefold()
    text_signal = any(
        marker in lowered
        for marker in (
            "text_evidence",
            "visible_text",
            "pseudo-text",
            "pseudo text",
            "readable",
            "typography",
        )
    )
    return {
        "decision": "retry",
        "claim_alignment": 0.0,
        "semantic_alignment": 0.0,
        "setup_alignment": 0.0,
        "coherent_scene": False,
        "visible_text": text_signal,
        "malformed_subject": False,
        "generic_architecture": False,
        "collage_layout": False,
        "reason": (
            "review transport was malformed or truncated; frame cannot be approved; "
            f"response_sha256={digest}"
        ),
        "repair_instruction": (
            "Regenerate one coherent topic-specific action scene with completely unmarked surfaces and no pseudo-text"
            if text_signal
            else "Regenerate one coherent topic-specific action scene with clear physical motion and stable geometry"
        ),
        "review_transport_recovered": True,
        "review_transport_sha256": digest,
    }


def extract_visual_review_payload_v60(text: str) -> dict[str, Any]:
    """Use strict v49 JSON when possible; convert only malformed transport into a retry.

    Quality thresholds remain owned by the semantic reviewer.  This function changes only the
    serialization failure mode: malformed output is equivalent to a rejected frame, never an
    approved frame and never an uncaught infrastructure exception.
    """
    from .production_visual_quality import VisualQualityError
    from .production_visual_review_json_v49 import extract_visual_review_json_v49

    try:
        return extract_visual_review_json_v49(text)
    except VisualQualityError as exc:
        message = str(exc)
        if not message.startswith("Visual reviewer returned malformed JSON"):
            raise
        return _transport_retry_payload(text)


def install_production_visual_review_transport_v60() -> None:
    """Install the final reviewer transport boundary used by the live semantic reviewer."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_visual_quality

    production_visual_quality._extract_json = extract_visual_review_payload_v60
    _INSTALLED = True
