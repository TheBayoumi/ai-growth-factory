from __future__ import annotations

import math
import os

from .production_voice_convergence_v28 import _SENTENCE_RE, _env_int, _split_long_unit


_INSTALLED = False


def voice_segment_capacity_v29(
    *,
    total_words: int,
    target_segments: int,
    minimum_words: int,
) -> int:
    """Derive safe voice-segment capacity from narration size without an arbitrary 12-part cap.

    The capacity is the greatest number of minimum-sized segments the transcript can legitimately
    require, bounded by an explicit operational ceiling. Final pace, tempo correction, fidelity,
    and perceptual review remain independent fail-closed gates.
    """
    if total_words < 1:
        raise ValueError("total_words must be positive")
    if minimum_words < 1:
        raise ValueError("minimum_words must be positive")
    requested = max(1, int(target_segments))
    absolute_maximum = _env_int("V29_ABSOLUTE_MAX_VOICE_SEGMENTS", 20, 8, 32)
    transcript_capacity = max(requested, math.ceil(total_words / minimum_words))
    return min(absolute_maximum, transcript_capacity)


def split_narration_for_voice_v29(text: str, target_segments: int) -> list[str]:
    """Create sentence-aligned TTS segments with duration-adaptive operational capacity.

    Short standalone sentences are merged before synthesis. Qwen adds a largely fixed onset and
    ending cadence to every generated file; measuring a ten-word technical sentence as an
    independent segment therefore produces structurally low WPM even when the spoken delivery is
    already as fast as the model can generate. A twelve-word default amortizes that fixed overhead
    without allowing long, unstable segments.
    """
    clean = " ".join(text.split()).strip()
    if not clean:
        raise ValueError("Narration is empty")

    minimum_words = _env_int("V28_MIN_VOICE_SEGMENT_WORDS", 12, 3, 20)
    maximum_words = _env_int("V28_MAX_VOICE_SEGMENT_WORDS", 24, minimum_words, 40)
    total_words = len(clean.split())
    capacity = voice_segment_capacity_v29(
        total_words=total_words,
        target_segments=target_segments,
        minimum_words=minimum_words,
    )

    sentences = [part.strip() for part in _SENTENCE_RE.split(clean) if part.strip()] or [clean]
    segments: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) <= maximum_words:
            segments.append(sentence)
        else:
            segments.extend(_split_long_unit(sentence, maximum_words))

    index = 0
    while index < len(segments):
        if len(segments[index].split()) >= minimum_words or len(segments) == 1:
            index += 1
            continue
        merged = False
        if index + 1 < len(segments):
            candidate = f"{segments[index]} {segments[index + 1]}"
            if len(candidate.split()) <= maximum_words:
                segments[index] = candidate
                del segments[index + 1]
                merged = True
        if not merged and index > 0:
            candidate = f"{segments[index - 1]} {segments[index]}"
            if len(candidate.split()) <= maximum_words:
                segments[index - 1] = candidate
                del segments[index]
                merged = True
                index -= 1
        if not merged:
            index += 1

    requested = max(1, int(target_segments))
    while len(segments) < requested:
        candidates = [
            (len(segment.split()), segment_index)
            for segment_index, segment in enumerate(segments)
            if len(segment.split()) >= minimum_words * 2
        ]
        if not candidates:
            break
        _size, segment_index = max(candidates)
        words = segments[segment_index].split()
        split_at = len(words) // 2
        segments[segment_index : segment_index + 1] = [
            " ".join(words[:split_at]),
            " ".join(words[split_at:]),
        ]

    if len(segments) > capacity:
        absolute_maximum = _env_int("V29_ABSOLUTE_MAX_VOICE_SEGMENTS", 20, 8, 32)
        raise ValueError(
            f"Narration requires {len(segments)} bounded TTS segments, above adaptive capacity "
            f"{capacity} for {total_words} words (absolute maximum {absolute_maximum})"
        )
    if any(not segment.strip() or len(segment.split()) > maximum_words for segment in segments):
        raise ValueError("Narration segmentation violated the v29 word-count contract")
    if " ".join(" ".join(segments).split()) != clean:
        raise ValueError("Narration segmentation changed the supplied transcript")
    return segments


def install_production_voice_capacity_v29() -> None:
    """Install adaptive segmentation after v28 convergence while preserving all voice gates."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import voice_pipeline

    voice_pipeline.split_narration = split_narration_for_voice_v29
    os.environ.setdefault("V29_ABSOLUTE_MAX_VOICE_SEGMENTS", "20")
    _INSTALLED = True
