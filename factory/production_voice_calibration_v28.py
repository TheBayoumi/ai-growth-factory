from __future__ import annotations

import math
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import NarrationSegment, VoiceContract


_INSTALLED = False
_SILENCE_THRESHOLD_DBFS = -42.0
_WINDOW_MS = 20
_MAX_INTERNAL_SILENCE_MS = 220
_MAX_EDGE_SILENCE_MS = 80


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


def install_production_voice_calibration_v28() -> None:
    """Compact dead air before conservative segment and track tempo correction."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import voice_pipeline

    original_pace = voice_pipeline._pace_correct_segment_assets

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

    voice_pipeline._pace_correct_segment_assets = calibrated_pace_correct_segment_assets
    _INSTALLED = True
