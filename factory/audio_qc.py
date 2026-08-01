from __future__ import annotations

import json
import math
import re
import subprocess
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg

from .config import Settings
from .models import AudioMetrics, NarrationSegment


class AudioQCError(RuntimeError):
    pass


MIN_TEMPO_FACTOR = 0.85
MAX_TEMPO_FACTOR = 1.15


def split_narration(text: str, target_segments: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        raise AudioQCError("Narration is empty")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if len(sentences) < target_segments:
        clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", clean) if part.strip()]
        if len(clauses) > len(sentences):
            sentences = clauses
    target_segments = min(max(1, target_segments), len(sentences))
    if target_segments == 1:
        return [clean]
    total_words = sum(len(sentence.split()) for sentence in sentences)
    segments: list[str] = []
    current: list[str] = []
    current_words = 0
    assigned_words = 0
    for index, sentence in enumerate(sentences):
        words = len(sentence.split())
        current.append(sentence)
        current_words += words
        remaining_sentences = len(sentences) - index - 1
        remaining_slots = target_segments - len(segments) - 1
        remaining_words = total_words - assigned_words
        ideal = remaining_words / max(remaining_slots + 1, 1)
        force_close = remaining_sentences == remaining_slots
        if remaining_slots > 0 and (current_words >= ideal or force_close):
            segments.append(" ".join(current))
            assigned_words += current_words
            current = []
            current_words = 0
    if current:
        segments.append(" ".join(current))
    if len(segments) != target_segments:
        raise AudioQCError(
            f"Could not split narration into {target_segments} segments; produced {len(segments)}"
        )
    return segments


def _read_pcm16(path: Path) -> tuple[array, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        if sample_width != 2:
            raise AudioQCError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit: {path}")
        samples = array("h")
        samples.frombytes(wav.readframes(frames))
    if channels > 1:
        mono = array("h")
        for offset in range(0, len(samples), channels):
            frame = samples[offset : offset + channels]
            mono.append(int(sum(frame) / len(frame)))
        samples = mono
        channels = 1
    return samples, sample_rate, channels


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def tempo_correction_factor(
    *,
    estimated_wpm: float,
    target_wpm: int,
    tolerance: int,
    minimum_factor: float = MIN_TEMPO_FACTOR,
    maximum_factor: float = MAX_TEMPO_FACTOR,
) -> float | None:
    """Return a bounded tempo factor only when it can bring pace inside the gate."""
    if estimated_wpm <= 0:
        raise AudioQCError("estimated_wpm must be positive")
    if target_wpm <= 0 or tolerance < 0:
        raise AudioQCError("target_wpm must be positive and tolerance must be non-negative")
    if not 0.5 <= minimum_factor <= 1.0 <= maximum_factor <= 2.0:
        raise AudioQCError("tempo correction bounds must straddle 1.0")

    lower_wpm = target_wpm - tolerance
    upper_wpm = target_wpm + tolerance
    if lower_wpm <= estimated_wpm <= upper_wpm:
        return None

    requested = target_wpm / estimated_wpm
    factor = min(max(requested, minimum_factor), maximum_factor)
    projected_wpm = estimated_wpm * factor
    if lower_wpm <= projected_wpm <= upper_wpm:
        return factor
    return None


def correct_audio_tempo(input_path: Path, output_path: Path, *, factor: float) -> Path:
    """Apply a conservative, pitch-preserving tempo correction with FFmpeg atempo."""
    if not MIN_TEMPO_FACTOR <= factor <= MAX_TEMPO_FACTOR:
        raise AudioQCError(
            f"Tempo factor must be between {MIN_TEMPO_FACTOR:.2f} and {MAX_TEMPO_FACTOR:.2f}"
        )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"atempo={factor:.6f}",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 44:
        raise AudioQCError(f"Audio tempo correction failed: {completed.stderr[-2000:]}")
    return output_path


def analyze_audio(
    path: Path,
    *,
    narration: str,
    settings: Settings,
    target_wpm: int | None = None,
) -> AudioMetrics:
    samples, sample_rate, channels = _read_pcm16(path)
    if not samples:
        raise AudioQCError("WAV contains no samples")
    full_scale = 32768.0
    normalized = [sample / full_scale for sample in samples]
    peak = max(abs(value) for value in normalized)
    rms = math.sqrt(sum(value * value for value in normalized) / len(normalized))
    peak_dbfs = 20 * math.log10(max(peak, 1e-12))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    clipping_ratio = sum(abs(value) >= 0.999 for value in normalized) / len(normalized)
    dc_offset = sum(normalized) / len(normalized)
    silence_threshold = 10 ** (-42 / 20)
    window_samples = max(1, int(sample_rate * 0.020))
    silent_windows: list[bool] = []
    for start in range(0, len(normalized), window_samples):
        window = normalized[start : start + window_samples]
        window_rms = math.sqrt(sum(value * value for value in window) / max(len(window), 1))
        silent_windows.append(window_rms < silence_threshold)
    silence_ratio = sum(silent_windows) / max(len(silent_windows), 1)
    longest = 0
    current = 0
    for is_silent in silent_windows:
        if is_silent:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    max_silence_seconds = longest * window_samples / sample_rate
    duration = len(normalized) / sample_rate
    estimated_wpm = len(narration.split()) / max(duration, 0.001) * 60
    target_wpm = target_wpm or settings.voice_contract.target_wpm
    failures: list[str] = []
    if peak_dbfs > settings.audio_peak_limit_dbfs + 0.05:
        failures.append(
            f"peak {peak_dbfs:.2f} dBFS exceeds limit {settings.audio_peak_limit_dbfs:.2f} dBFS"
        )
    if rms_dbfs < settings.audio_min_rms_dbfs:
        failures.append(
            f"RMS {rms_dbfs:.2f} dBFS is below minimum {settings.audio_min_rms_dbfs:.2f} dBFS"
        )
    if clipping_ratio > settings.audio_max_clipping_ratio:
        failures.append(
            f"clipping ratio {clipping_ratio:.6f} exceeds {settings.audio_max_clipping_ratio:.6f}"
        )
    if silence_ratio > settings.audio_max_silence_ratio:
        failures.append(
            f"silence ratio {silence_ratio:.3f} exceeds {settings.audio_max_silence_ratio:.3f}"
        )
    if max_silence_seconds > settings.audio_max_silence_seconds:
        failures.append(
            f"longest silence {max_silence_seconds:.2f}s exceeds {settings.audio_max_silence_seconds:.2f}s"
        )
    if abs(estimated_wpm - target_wpm) > settings.audio_wpm_tolerance:
        failures.append(
            f"estimated pace {estimated_wpm:.1f} WPM is outside target {target_wpm}±{settings.audio_wpm_tolerance}"
        )
    if abs(dc_offset) > 0.02:
        failures.append(f"DC offset {dc_offset:.4f} exceeds 0.0200")
    return AudioMetrics(
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        max_silence_seconds=max_silence_seconds,
        estimated_wpm=estimated_wpm,
        dc_offset=dc_offset,
        passed=not failures,
        failures=tuple(failures),
    )


def concatenate_segments(
    segments: Iterable[NarrationSegment],
    output: Path,
    *,
    pause_ms: int,
) -> tuple[Path, list[NarrationSegment]]:
    ordered = sorted(segments, key=lambda segment: segment.segment_id)
    if not ordered:
        raise AudioQCError("No narration segments were supplied")
    sample_rate: int | None = None
    all_samples = array("h")
    timed: list[NarrationSegment] = []
    cursor = 0.0
    for index, segment in enumerate(ordered):
        samples, rate, channels = _read_pcm16(segment.audio_path)
        if channels != 1:
            raise AudioQCError("Narration segments must be mono")
        if sample_rate is None:
            sample_rate = rate
        elif sample_rate != rate:
            raise AudioQCError("Narration segments have inconsistent sample rates")
        start = cursor
        all_samples.extend(samples)
        cursor += len(samples) / rate
        end = cursor
        timed.append(replace(segment, start_seconds=start, end_seconds=end))
        if index + 1 < len(ordered) and pause_ms > 0:
            pause_samples = int(rate * pause_ms / 1000)
            all_samples.extend(array("h", [0]) * pause_samples)
            cursor += pause_samples / rate
    assert sample_rate is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(all_samples.tobytes())
    return output, timed


def normalize_audio(input_path: Path, output_path: Path, *, target_lufs: float, peak_dbfs: float) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={peak_dbfs}:LRA=11",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise AudioQCError(f"Audio normalization failed: {completed.stderr[-2000:]}")
    return output_path


def convert_for_reviewer(input_path: Path, output_path: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise AudioQCError(f"Reviewer conversion failed: {completed.stderr[-2000:]}")
    return output_path


def write_manifest(
    output: Path,
    *,
    segments: list[NarrationSegment],
    metrics: AudioMetrics,
    attempts: int,
    reviews: list[dict[str, object]],
    voice_contract: dict[str, object] | None = None,
    generator: dict[str, object] | None = None,
    reviewer: dict[str, object] | None = None,
) -> Path:
    data = {
        "schema_version": 2,
        "attempts": attempts,
        "generator": generator,
        "reviewer": reviewer,
        "voice_contract": voice_contract,
        "metrics": metrics.as_dict(),
        "segments": [
            {
                "segment_id": segment.segment_id,
                "text": segment.text,
                "instruction": segment.instruction,
                "audio_path": str(segment.audio_path),
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "attempt": segment.attempt,
            }
            for segment in segments
        ],
        "reviews": reviews,
    }
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
