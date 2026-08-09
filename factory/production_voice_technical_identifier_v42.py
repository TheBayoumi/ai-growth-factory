from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any

from .video_profile import VideoProfile


_INSTALLED = False
_NUMERIC_TOKEN_RE = re.compile(
    r"^(?P<currency>\$)?(?P<sign>[+-])?"
    r"(?P<integer>(?:\d{1,3}(?:,\d{3})+|\d+))"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?P<percent>%?)(?P<suffix_plus>\+?)$"
)
_ACTIVE_PACE_MULTIPLIER: ContextVar[float] = ContextVar(
    "voice_technical_identifier_pace_multiplier_v42",
    default=1.0,
)


def _integer_spoken_units_v42(value: int) -> int:
    """Conservatively count English cardinal speech units for a non-negative integer."""
    value = abs(int(value))
    if value < 20:
        return 1
    if value < 100:
        return 1 if value % 10 == 0 else 2
    if value < 1_000:
        remainder = value % 100
        return 2 if remainder == 0 else 2 + _integer_spoken_units_v42(remainder)
    for scale in (1_000_000_000, 1_000_000, 1_000):
        if value >= scale:
            leading, remainder = divmod(value, scale)
            units = _integer_spoken_units_v42(leading) + 1
            if remainder:
                units += _integer_spoken_units_v42(remainder)
            return min(8, units)
    return 1


def _numeric_token_weight_v42(value: str) -> float | None:
    match = _NUMERIC_TOKEN_RE.fullmatch(value)
    if match is None:
        return None

    integer = int(match.group("integer").replace(",", ""))
    units = _integer_spoken_units_v42(integer)
    fraction = match.group("fraction") or ""
    if fraction:
        # Qwen normally speaks decimals digit-by-digit after the word "point".
        units += 1 + len(fraction)
    if match.group("currency"):
        units += 1  # dollar / dollars
    if match.group("sign"):
        units += 1  # plus / minus
    if match.group("percent"):
        units += 1
    if match.group("suffix_plus"):
        units += 1
    # Keep one unusual numeric token from dominating a segment-level pace decision.
    return float(min(8, max(1, units)))


def technical_token_weight_v42(token: str) -> float:
    """Estimate deterministic spoken word-equivalents for technical and numeric tokens.

    Plain words and acronyms remain one publication word. Mixed identifiers such as
    ``LFM2.5-2.6B`` receive bounded extra weight for separately spoken groups. Numeric tokens
    are counted from their spoken cardinal form, so ``81`` is two units and ``98.6%`` is five
    ("ninety eight point six percent"). This corrects the measurement denominator only; it does
    not change the configured WPM window or tempo ceiling.
    """
    value = token.strip(".,;:!?()[]{}\"'")
    if not value:
        return 0.0

    numeric = _numeric_token_weight_v42(value)
    if numeric is not None:
        return numeric

    has_letter = any(character.isalpha() for character in value)
    has_digit = any(character.isdigit() for character in value)
    if not (has_letter and has_digit):
        return 1.0
    groups = re.findall(r"[A-Za-z]+|\d+", value)
    return min(3.0, 1.0 + 0.5 * max(1, len(groups) - 1))


def speech_equivalent_word_count_v42(text: str) -> float:
    # Whitespace tokens preserve suffixes such as %, +, and comma-grouped numerals that the old
    # alphanumeric regex discarded before technical_token_weight_v42 could measure them.
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(technical_token_weight_v42(token) for token in tokens)


def written_word_count_v42(text: str) -> int:
    return len(text.split())


def pace_multiplier_v42(text: str) -> float:
    written = max(1, written_word_count_v42(text))
    return speech_equivalent_word_count_v42(text) / written


def speech_equivalent_wpm_v42(text: str, duration_seconds: float) -> float:
    return speech_equivalent_word_count_v42(text) / max(duration_seconds, 0.001) * 60.0


def _effective_observation(observed_wpm: float) -> float:
    return float(observed_wpm) * _ACTIVE_PACE_MULTIPLIER.get()


def install_production_voice_technical_identifier_v42() -> None:
    """Make technical/numeric speech pace measurement auditable and fail-closed."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc, production_voice_calibration_v28 as calibration
    from . import production_voice_convergence_v28 as convergence
    from . import voice_pipeline

    profile = VideoProfile.from_env()
    base_tts = voice_pipeline.Qwen3TTS
    original_reachable = calibration.segment_candidate_reachable_v28
    original_synthesis_target = calibration.synthesis_target_for_observation_v28
    original_analyze_audio = audio_qc.analyze_audio

    def technical_identifier_reachable(
        observed_wpm: float,
        *,
        profile: VideoProfile,
        minimum_tempo_factor: float = 0.85,
    ) -> bool:
        return original_reachable(
            _effective_observation(observed_wpm),
            profile=profile,
            minimum_tempo_factor=minimum_tempo_factor,
        )

    def technical_identifier_candidate_score(
        observed_wpm: float,
        *,
        profile: VideoProfile,
        minimum_tempo_factor: float,
    ) -> tuple[float, float, float]:
        effective = max(0.001, _effective_observation(observed_wpm))
        required = profile.target_wpm / effective
        factor = min(max(required, minimum_tempo_factor), profile.maximum_tempo_factor)
        projected = effective * factor
        reachable = original_reachable(
            effective,
            profile=profile,
            minimum_tempo_factor=minimum_tempo_factor,
        )
        penalty = 0.0 if reachable else 100.0
        score = penalty + abs(projected - profile.target_wpm) + abs(factor - 1.0) * 4.0
        return score, factor, projected

    def technical_identifier_synthesis_target(
        observed_wpm: float,
        *,
        profile: VideoProfile,
    ) -> int:
        return original_synthesis_target(
            _effective_observation(observed_wpm),
            profile=profile,
        )

    class TechnicalIdentifierAwareQwen3TTS(base_tts):
        def generate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
        ) -> Path:
            written = max(1, written_word_count_v42(text))
            equivalent = speech_equivalent_word_count_v42(text)
            multiplier = equivalent / written
            event_start = len(calibration._CALIBRATION_EVENTS)
            token = _ACTIVE_PACE_MULTIPLIER.set(multiplier)
            try:
                return super().generate(
                    text=text,
                    instruction=instruction,
                    output_path=output_path,
                    seed=seed,
                )
            finally:
                _ACTIVE_PACE_MULTIPLIER.reset(token)
                for event in calibration._CALIBRATION_EVENTS[event_start:]:
                    if event.get("type") != "measured_tts_calibration_v28":
                        continue
                    raw = float(event.get("observed_compacted_wpm") or 0.0)
                    event["written_word_count"] = written
                    event["speech_equivalent_word_count"] = round(equivalent, 3)
                    event["technical_identifier_pace_multiplier"] = round(multiplier, 6)
                    event["observed_written_wpm"] = round(raw, 3)
                    event["observed_compacted_wpm"] = round(raw * multiplier, 3)
                    event["word_count"] = round(equivalent, 3)

    def technical_segment_wpm(segment: Any) -> float:
        return speech_equivalent_wpm_v42(
            segment.text,
            audio_qc.wav_duration(segment.audio_path),
        )

    def technical_analyze_audio(
        path: Path,
        *,
        narration: str,
        settings: Any,
        target_wpm: int | None = None,
    ) -> Any:
        metrics = original_analyze_audio(
            path,
            narration=narration,
            settings=settings,
            target_wpm=target_wpm,
        )
        adjusted_wpm = speech_equivalent_wpm_v42(
            narration,
            metrics.duration_seconds,
        )
        resolved_target = target_wpm or settings.voice_contract.target_wpm
        failures = [
            failure
            for failure in metrics.failures
            if not str(failure).startswith("estimated pace ")
        ]
        if abs(adjusted_wpm - resolved_target) > settings.audio_wpm_tolerance:
            failures.append(
                f"estimated pace {adjusted_wpm:.1f} WPM is outside target "
                f"{resolved_target}±{settings.audio_wpm_tolerance}"
            )
        return replace(
            metrics,
            estimated_wpm=adjusted_wpm,
            failures=tuple(failures),
            passed=not failures,
        )

    calibration.segment_candidate_reachable_v28 = technical_identifier_reachable
    calibration._candidate_score = technical_identifier_candidate_score
    calibration.synthesis_target_for_observation_v28 = technical_identifier_synthesis_target
    convergence.segment_wpm_v28 = technical_segment_wpm
    audio_qc.analyze_audio = technical_analyze_audio
    voice_pipeline.analyze_audio = technical_analyze_audio
    voice_pipeline.Qwen3TTS = TechnicalIdentifierAwareQwen3TTS
    _INSTALLED = True
