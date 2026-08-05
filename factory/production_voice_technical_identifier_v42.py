from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any

from .video_profile import VideoProfile


_INSTALLED = False
_TECHNICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
_ACTIVE_PACE_MULTIPLIER: ContextVar[float] = ContextVar(
    "voice_technical_identifier_pace_multiplier_v42",
    default=1.0,
)


def technical_token_weight_v42(token: str) -> float:
    """Estimate spoken word-equivalents only for mixed letter/number identifiers.

    Plain acronyms such as AI remain one publication word. Identifiers such as LFM2.5-2.6B
    receive bounded additional weight because the synthesizer speaks their letter and number
    groups separately. The cap prevents a long identifier from weakening the pace gate.
    """
    value = token.strip(".,;:!?()[]{}\"'")
    has_letter = any(character.isalpha() for character in value)
    has_digit = any(character.isdigit() for character in value)
    if not (has_letter and has_digit):
        return 1.0
    groups = re.findall(r"[A-Za-z]+|\d+", value)
    return min(3.0, 1.0 + 0.5 * max(1, len(groups) - 1))


def speech_equivalent_word_count_v42(text: str) -> float:
    tokens = _TECHNICAL_TOKEN_RE.findall(text)
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
    """Make technical-identifier pace measurement auditable and fail-closed."""
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
