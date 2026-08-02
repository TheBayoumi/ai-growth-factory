from __future__ import annotations

import re
from typing import Any


_INSTALLED = False
_REPAIR_INSTRUCTIONS = {
    "script_fidelity": "Speak every supplied word exactly once, with no omissions, additions, substitutions, or paraphrasing.",
    "naturalness": "Regenerate with natural human phrasing, smooth articulation, and no robotic cadence or synthetic strain.",
    "authority": "Use a confident, credible technology-news delivery without sounding theatrical or exaggerated.",
    "engagement": "Increase vocal variation and emphasis on the key factual words while preserving a professional delivery.",
    "pronunciation": "Pronounce every technical term clearly and consistently, preserving the supplied transcript exactly.",
    "pace": "Match the requested short-form pace with steady timing and no rushed or dragged phrases.",
    "pause_quality": "Use short natural pauses at clause boundaries and remove awkward silence or clipped joins.",
    "emotional_match": "Match the requested energy and warmth while remaining factual and non-sensational.",
    "audio_artifacts": "Regenerate clean audio with no noise, clicks, distortion, metallic resonance, or boundary artifacts.",
}
_PLACEHOLDER_PATTERNS = (
    re.compile(r"standalone\s+qwen3[- ]tts\s+repair\s+instruction", re.IGNORECASE),
    re.compile(r"specific\s+audible\s+defect", re.IGNORECASE),
    re.compile(r"or\s+empty\s+when\s+approved", re.IGNORECASE),
    re.compile(r"insert\s+(?:a\s+)?(?:reason|instruction)\s+here", re.IGNORECASE),
)


def _is_placeholder(value: str) -> bool:
    clean = " ".join(value.split()).strip()
    if not clean:
        return True
    return any(pattern.search(clean) for pattern in _PLACEHOLDER_PATTERNS)


def normalize_retry_feedback(data: dict[str, Any]) -> dict[str, Any]:
    """Complete incomplete or schema-echoed Qwen retry feedback from its scores.

    The model still decides whether the segment needs retry. This function only replaces
    missing fields or literal schema placeholders with an actionable instruction derived
    from the returned score vector. It never changes decisions, scores, segment IDs, or
    the bounded attempt count.
    """
    if str(data.get("decision") or "").strip().lower() != "retry":
        return data
    reason = str(data.get("reason") or "").strip()
    instruction = str(data.get("tts_instruction") or "").strip()
    reason_missing = _is_placeholder(reason)
    instruction_missing = _is_placeholder(instruction)
    if not reason_missing and not instruction_missing:
        return data

    raw_scores = data.get("scores")
    scores = dict(raw_scores) if isinstance(raw_scores, dict) else {}
    candidates: list[tuple[float, str]] = []
    for field in _REPAIR_INSTRUCTIONS:
        try:
            value = float(scores.get(field, 1.0))
        except (TypeError, ValueError):
            value = 1.0
        candidates.append((value, field))
    lowest_value, lowest_field = min(candidates, default=(0.0, "script_fidelity"))

    corrected = dict(data)
    if reason_missing:
        corrected["reason"] = (
            "Reviewer requested a segment retry; the lowest returned quality score was "
            f"{lowest_field} at {lowest_value:.2f}."
        )
    if instruction_missing:
        corrected["tts_instruction"] = _REPAIR_INSTRUCTIONS[lowest_field]
    return corrected


def install_production_reviewer_feedback() -> None:
    """Install deterministic completion for incomplete Qwen retry feedback."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import qwen_omni_reviewer

    original = qwen_omni_reviewer._normalize_segment_result

    def normalize(data: dict[str, Any], segment_id: int) -> dict[str, Any]:
        return original(normalize_retry_feedback(data), segment_id)

    qwen_omni_reviewer._normalize_segment_result = normalize
    _INSTALLED = True
