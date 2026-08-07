from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import NarrationSegment, VideoPackage
from .production_voice_technical_identifier_v42 import (
    speech_equivalent_word_count_v42,
)
from .video_profile import VideoProfile


_INSTALLED = False
_SPOKEN_EQUIVALENT_MAX = 140.0
_GENERATION_TARGET_MIN_WORDS = 130
_GENERATION_TARGET_MAX_WORDS = 134
_DURATION_RULES = """
SPOKEN-DURATION CAPACITY:
- Keep the existing hard narration length of 130-140 written words.
- Prefer 130-134 written words so natural pauses remain inside the 55-62 second video window.
- The finished narration must contain at most 140 spoken-word equivalents.
- Numeric tokens count by how they are spoken: 81 is about 2 words; 98.6% about 5; 500,000+ about 4.
- Preserve source-backed measurements, but remove repetition and low-information filler before adding extra numbers.
""".strip()


def spoken_equivalent_count_v51(text: str) -> float:
    return float(speech_equivalent_word_count_v42(text))


def validate_spoken_duration_capacity_v51(
    package: VideoPackage,
    *,
    maximum_spoken_equivalents: float = _SPOKEN_EQUIVALENT_MAX,
) -> float:
    """Reject scripts whose spoken form cannot naturally fit the frozen short-form window."""
    from .local_llm import LocalLLMError

    written = len(package.narration.split())
    spoken = spoken_equivalent_count_v51(package.narration)
    if spoken > maximum_spoken_equivalents + 1e-9:
        raise LocalLLMError(
            "Production narration exceeds spoken-duration budget: "
            f"{spoken:.1f} spoken-word equivalents; maximum "
            f"{maximum_spoken_equivalents:.0f}. Keep the existing 130-140 written-word "
            "contract, rewrite toward 130-134 written words, preserve the strongest "
            "source-backed measurements, and remove repetition or low-information filler. "
            "Numeric tokens count by spoken form (81≈2, 98.6%≈5, 500,000+≈4). "
            f"Current written words: {written}."
        )
    return spoken


def _projected_pause_seconds_v51(profile: VideoProfile, segment_count: int) -> float:
    if segment_count <= 1:
        return 0.0
    regular_count = max(0, segment_count - 2)
    return (
        regular_count * profile.segment_pause_ms / 1000.0
        + profile.pre_cta_pause_ms / 1000.0
    )


def projected_narration_segments_v51(
    settings: Any,
    package: VideoPackage,
    profile: VideoProfile,
) -> tuple[tuple[NarrationSegment, ...], float]:
    """Project duration using the same spoken-equivalent denominator as live TTS QC."""
    from . import voice_pipeline
    from .production_editorial_v28 import ProductionPreflightError

    texts = voice_pipeline.split_narration(
        package.narration,
        settings.narration_segments,
    )
    if not texts:
        raise ProductionPreflightError("Narration produced no projected voice segments")

    spoken_counts = [max(1.0, spoken_equivalent_count_v51(text)) for text in texts]
    speech_seconds = sum(spoken_counts) * 60.0 / profile.target_wpm
    pause_seconds = _projected_pause_seconds_v51(profile, len(texts))
    total_duration = speech_seconds + pause_seconds
    if not profile.minimum_video_seconds <= total_duration <= profile.maximum_video_seconds:
        raise ProductionPreflightError(
            f"Projected spoken-equivalent narration duration {total_duration:.3f}s is outside "
            f"the frozen production window {profile.minimum_video_seconds:.1f}-"
            f"{profile.maximum_video_seconds:.1f}s; spoken_word_equivalents="
            f"{sum(spoken_counts):.1f}, written_words={len(package.narration.split())}"
        )

    pauses = [profile.segment_pause_ms / 1000.0] * max(0, len(texts) - 1)
    if pauses:
        pauses[-1] = profile.pre_cta_pause_ms / 1000.0

    cursor = 0.0
    segments: list[NarrationSegment] = []
    for index, (text, spoken_count) in enumerate(zip(texts, spoken_counts, strict=True)):
        duration = spoken_count * 60.0 / profile.target_wpm
        start = cursor
        end = start + duration
        segments.append(
            NarrationSegment(
                segment_id=index,
                text=text,
                instruction="spoken-equivalent deterministic preflight timing v51",
                audio_path=Path("preflight-no-audio.wav"),
                start_seconds=round(start, 6),
                end_seconds=round(end, 6),
            )
        )
        cursor = end + (pauses[index] if index < len(pauses) else 0.0)
    return tuple(segments), round(total_duration, 6)


def maximum_spoken_equivalents_for_profile_v51(
    profile: VideoProfile,
    *,
    segment_count: int = 8,
) -> int:
    """Expose the mathematical ceiling used to justify the conservative 140-word budget."""
    available_speech = max(
        0.0,
        profile.maximum_video_seconds
        - _projected_pause_seconds_v51(profile, segment_count),
    )
    return math.floor(available_speech * profile.target_wpm / 60.0)


def install_production_spoken_duration_capacity_v51() -> None:
    """Move duration convergence ahead of TTS without relaxing any publication threshold."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm, production_content, production_editorial_v28

    # v46 installs first and aligns the legacy repair path. v51 narrows the preferred generation
    # range only; the hard production validator remains 130-140 written words.
    local_llm.NARRATION_TARGET_MIN_WORDS = _GENERATION_TARGET_MIN_WORDS
    local_llm.NARRATION_TARGET_MAX_WORDS = _GENERATION_TARGET_MAX_WORDS

    current_rules = production_content._RULES
    if _DURATION_RULES not in current_rules:
        production_content._RULES = current_rules + "\n" + _DURATION_RULES

    original_validate = production_content._validate_publishable_content

    def validate_publishable_with_duration(
        package: VideoPackage,
        sources: list[Any],
    ) -> None:
        original_validate(package, sources)
        validate_spoken_duration_capacity_v51(package)

    production_content._validate_publishable_content = validate_publishable_with_duration
    production_editorial_v28._projected_narration_segments = projected_narration_segments_v51
    _INSTALLED = True
