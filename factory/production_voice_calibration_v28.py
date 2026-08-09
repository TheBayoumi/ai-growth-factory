from __future__ import annotations

import math
import shutil
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import NarrationSegment, VoiceContract
from .video_profile import VideoProfile


_INSTALLED = False
_SILENCE_THRESHOLD_DBFS = -42.0
_WINDOW_MS = 20
_MAX_INTERNAL_SILENCE_MS = 220
_MAX_EDGE_SILENCE_MS = 80
_MAX_INTERNAL_TTS_ATTEMPTS = 3
_CALIBRATION_EVENTS: list[dict[str, object]] = []


def _read_mono_pcm16(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        samples = array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV: {path}")
    if channels > 1:
        mono = array("h")
        for offset in range(0, len(samples), channels):
            frame = samples[offset : offset + channels]
            mono.append(int(sum(frame) / max(1, len(frame))))
        samples = mono
    return samples, sample_rate


def _write_mono_pcm16(path: Path, samples: array, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def _window_is_silent(samples: array, threshold: float) -> bool:
    if not samples:
        return True
    rms = math.sqrt(sum(int(value) * int(value) for value in samples) / len(samples))
    return rms < threshold


def compact_excess_silence_v28(
    input_path: Path,
    output_path: Path,
    *,
    threshold_dbfs: float = _SILENCE_THRESHOLD_DBFS,
    window_ms: int = _WINDOW_MS,
    maximum_internal_silence_ms: int = _MAX_INTERNAL_SILENCE_MS,
    maximum_edge_silence_ms: int = _MAX_EDGE_SILENCE_MS,
) -> dict[str, float | int | str]:
    """Shorten excessive pauses without changing any non-silent speech samples.

    Silence is measured in fixed PCM windows. Internal pauses retain a natural 220 ms by
    default; leading and trailing dead air retain 80 ms. For a shortened pause, samples from
    both edges are preserved so the waveform still enters and exits speech smoothly. No
    resampling, pitch shifting, or phoneme compression occurs in this stage.
    """
    if window_ms < 5 or window_ms > 100:
        raise ValueError("window_ms must be between 5 and 100")
    if maximum_internal_silence_ms < window_ms:
        raise ValueError("maximum_internal_silence_ms must be at least one window")
    if maximum_edge_silence_ms < 0:
        raise ValueError("maximum_edge_silence_ms must be non-negative")

    samples, sample_rate = _read_mono_pcm16(input_path)
    window_samples = max(1, round(sample_rate * window_ms / 1000.0))
    threshold = 32768.0 * (10.0 ** (threshold_dbfs / 20.0))
    windows: list[tuple[int, int, bool]] = []
    for start in range(0, len(samples), window_samples):
        end = min(len(samples), start + window_samples)
        windows.append((start, end, _window_is_silent(samples[start:end], threshold)))

    output = array("h")
    removed_samples = 0
    longest_before_samples = 0
    longest_after_samples = 0
    index = 0
    while index < len(windows):
        start, end, silent = windows[index]
        if not silent:
            output.extend(samples[start:end])
            index += 1
            continue

        run_start_index = index
        while index + 1 < len(windows) and windows[index + 1][2]:
            index += 1
        run_end_index = index
        run_start = windows[run_start_index][0]
        run_end = windows[run_end_index][1]
        run_samples = run_end - run_start
        longest_before_samples = max(longest_before_samples, run_samples)
        at_edge = run_start_index == 0 or run_end_index == len(windows) - 1
        keep_ms = maximum_edge_silence_ms if at_edge else maximum_internal_silence_ms
        keep_samples = min(run_samples, max(0, round(sample_rate * keep_ms / 1000.0)))
        longest_after_samples = max(longest_after_samples, keep_samples)

        if keep_samples >= run_samples:
            output.extend(samples[run_start:run_end])
        elif keep_samples > 0:
            first = keep_samples // 2
            second = keep_samples - first
            if first:
                output.extend(samples[run_start : run_start + first])
            if second:
                output.extend(samples[run_end - second : run_end])
            removed_samples += run_samples - keep_samples
        else:
            removed_samples += run_samples
        index += 1

    _write_mono_pcm16(output_path, output, sample_rate)
    before_seconds = len(samples) / max(1, sample_rate)
    after_seconds = len(output) / max(1, sample_rate)
    return {
        "type": "deterministic_silence_compaction_v28",
        "input_path": str(input_path),
        "audio_path": str(output_path),
        "before_seconds": round(before_seconds, 6),
        "after_seconds": round(after_seconds, 6),
        "removed_seconds": round(removed_samples / max(1, sample_rate), 6),
        "longest_silence_before_seconds": round(
            longest_before_samples / max(1, sample_rate), 6
        ),
        "longest_silence_after_seconds": round(
            longest_after_samples / max(1, sample_rate), 6
        ),
        "threshold_dbfs": threshold_dbfs,
        "window_ms": window_ms,
        "maximum_internal_silence_ms": maximum_internal_silence_ms,
        "maximum_edge_silence_ms": maximum_edge_silence_ms,
    }


def calibrated_segment_band_v28(profile: VideoProfile) -> tuple[float, float]:
    """Keep segments slightly tighter than the whole-track acceptance range."""
    lower = max(float(profile.minimum_wpm), float(profile.target_wpm - 2))
    upper = min(float(profile.maximum_wpm), float(profile.target_wpm + 2))
    if lower > upper:
        raise ValueError("The calibrated segment WPM band is empty")
    return lower, upper


def segment_candidate_reachable_v28(
    observed_wpm: float,
    *,
    profile: VideoProfile,
    minimum_tempo_factor: float = 0.85,
) -> bool:
    """Return whether conservative tempo correction can place a take in the segment band."""
    lower, upper = calibrated_segment_band_v28(profile)
    return (
        observed_wpm * profile.maximum_tempo_factor >= lower - 1e-6
        and observed_wpm * minimum_tempo_factor <= upper + 1e-6
    )


def synthesis_target_for_observation_v28(
    observed_wpm: float,
    *,
    profile: VideoProfile,
) -> int:
    """Calibrate Qwen's requested synthesis pace from its measured output bias."""
    observed = max(1.0, float(observed_wpm))
    projected = profile.target_wpm * (profile.target_wpm / observed)
    if observed < profile.target_wpm:
        return max(150, min(185, round(projected)))
    return max(115, min(138, round(projected)))


def _candidate_score(
    observed_wpm: float,
    *,
    profile: VideoProfile,
    minimum_tempo_factor: float,
) -> tuple[float, float, float]:
    required = profile.target_wpm / max(observed_wpm, 0.001)
    factor = min(max(required, minimum_tempo_factor), profile.maximum_tempo_factor)
    projected = observed_wpm * factor
    reachable = segment_candidate_reachable_v28(
        observed_wpm,
        profile=profile,
        minimum_tempo_factor=minimum_tempo_factor,
    )
    penalty = 0.0 if reachable else 100.0
    score = penalty + abs(projected - profile.target_wpm) + abs(factor - 1.0) * 4.0
    return score, factor, projected


def _calibration_instruction(
    base_instruction: str,
    *,
    observed_wpm: float,
    requested_wpm: int,
) -> str:
    direction = "brisker" if requested_wpm > observed_wpm else "slower"
    return (
        base_instruction.rstrip(" .")
        + ". The previous generated take measured "
        + f"{observed_wpm:.1f} words per minute after removing only excessive dead air. "
        + f"Regenerate with a {direction}, natural articulation targeting approximately "
        + f"{requested_wpm} words per minute before post-processing. Keep clause pauses "
        + "between roughly 120 and 220 milliseconds, pronounce every technical word clearly, "
        + "and do not omit, repeat, or paraphrase any supplied word. This is synthesis "
        + "calibration; final publication pace remains 138 to 146 words per minute."
    )


def install_production_voice_calibration_v28() -> None:
    """Calibrate each TTS take, compact dead air, and preserve audit evidence."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc, qwen_tts, voice_pipeline

    profile = VideoProfile.from_env()
    original_pace = voice_pipeline._pace_correct_segment_assets
    original_write_manifest = voice_pipeline.write_manifest

    class CalibratedQwen3TTS(qwen_tts.Qwen3TTS):
        def __init__(self, settings: Any) -> None:
            super().__init__(settings)
            _CALIBRATION_EVENTS.clear()

        def generate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
        ) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            minimum_factor = float(audio_qc.MIN_TEMPO_FACTOR)
            current_instruction = instruction
            candidates: list[tuple[float, Path, dict[str, object]]] = []

            for internal_attempt in range(1, _MAX_INTERNAL_TTS_ATTEMPTS + 1):
                candidate = output_path.with_name(
                    f"{output_path.stem}-calibration-{internal_attempt}{output_path.suffix}"
                )
                calibration_seed = (
                    int(seed) + (internal_attempt - 1) * 0x9E3779B1
                ) & 0xFFFFFFFF
                super().generate(
                    text=text,
                    instruction=current_instruction,
                    output_path=candidate,
                    seed=calibration_seed,
                )
                evaluation = candidate.with_name(
                    f"{candidate.stem}-silence-evaluation{candidate.suffix}"
                )
                compaction = compact_excess_silence_v28(candidate, evaluation)
                duration = float(compaction["after_seconds"])
                observed_wpm = len(text.split()) / max(duration, 0.001) * 60.0
                score, factor, projected = _candidate_score(
                    observed_wpm,
                    profile=profile,
                    minimum_tempo_factor=minimum_factor,
                )
                reachable = segment_candidate_reachable_v28(
                    observed_wpm,
                    profile=profile,
                    minimum_tempo_factor=minimum_factor,
                )
                event: dict[str, object] = {
                    "type": "measured_tts_calibration_v28",
                    "requested_output": str(output_path),
                    "candidate_path": str(candidate),
                    "internal_attempt": internal_attempt,
                    "seed": calibration_seed,
                    "word_count": len(text.split()),
                    "observed_compacted_wpm": round(observed_wpm, 3),
                    "required_tempo_factor": round(factor, 6),
                    "projected_wpm": round(projected, 3),
                    "reachable": reachable,
                    "removed_silence_seconds": compaction["removed_seconds"],
                    "selected": False,
                }
                candidates.append((score, candidate, event))
                evaluation.unlink(missing_ok=True)
                if reachable:
                    break
                requested_wpm = synthesis_target_for_observation_v28(
                    observed_wpm,
                    profile=profile,
                )
                event["next_requested_synthesis_wpm"] = requested_wpm
                current_instruction = _calibration_instruction(
                    instruction,
                    observed_wpm=observed_wpm,
                    requested_wpm=requested_wpm,
                )

            if not candidates:
                raise RuntimeError("Qwen TTS calibration produced no candidates")
            candidates.sort(key=lambda item: item[0])
            _score, selected_path, selected_event = candidates[0]
            selected_event["selected"] = True
            for _candidate_score_value, candidate_path, event in candidates:
                _CALIBRATION_EVENTS.append(event)
                if candidate_path == selected_path:
                    shutil.copy2(candidate_path, output_path)
                candidate_path.unlink(missing_ok=True)
            return output_path

    def calibrated_pace_correct_segment_assets(
        segments: list[NarrationSegment],
        *,
        workdir: Path,
        pipeline_attempt: int,
        contract: VoiceContract,
        settings: Any,
    ) -> tuple[list[NarrationSegment], list[dict[str, object]]]:
        compacted: list[NarrationSegment] = []
        events: list[dict[str, object]] = []
        for segment in segments:
            output = workdir / "segments" / (
                f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                f"silence-compacted-{pipeline_attempt}.wav"
            )
            event = compact_excess_silence_v28(segment.audio_path, output)
            event.update(
                {
                    "attempt": pipeline_attempt,
                    "segment_id": segment.segment_id,
                    "segment_attempt": segment.attempt,
                }
            )
            events.append(event)
            compacted.append(replace(segment, audio_path=output))

        corrected, pacing_events = original_pace(
            compacted,
            workdir=workdir,
            pipeline_attempt=pipeline_attempt,
            contract=contract,
            settings=settings,
        )
        return corrected, [*events, *pacing_events]

    def write_manifest_with_calibration(
        output: Path,
        *,
        segments: list[NarrationSegment],
        metrics: Any,
        attempts: int,
        reviews: list[dict[str, object]],
        voice_contract: dict[str, object] | None = None,
        generator: dict[str, object] | None = None,
        reviewer: dict[str, object] | None = None,
    ) -> Path:
        calibration = [dict(event) for event in _CALIBRATION_EVENTS]
        return original_write_manifest(
            output,
            segments=segments,
            metrics=metrics,
            attempts=attempts,
            reviews=[*calibration, *reviews],
            voice_contract=voice_contract,
            generator=generator,
            reviewer=reviewer,
        )

    voice_pipeline.Qwen3TTS = CalibratedQwen3TTS
    voice_pipeline._pace_correct_segment_assets = calibrated_pace_correct_segment_assets
    voice_pipeline.write_manifest = write_manifest_with_calibration
    _INSTALLED = True
