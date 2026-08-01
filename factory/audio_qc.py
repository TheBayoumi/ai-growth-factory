from __future__ import annotations

import json
import math
import subprocess
import wave
from array import array
from dataclasses import replace
from pathlib import Path

import imageio_ffmpeg

from .config import Settings
from .models import AudioMetrics, NarrationSegment


def split_narration(text: str, target_segments: int) -> list[str]:
    import re

    clean = re.sub(r"\s+", " ", text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if not sentences:
        raise RuntimeError("narration is empty")
    target = min(max(target_segments, 1), len(sentences))
    buckets = [[] for _ in range(target)]
    counts = [0] * target
    for sentence in sentences:
        index = min(range(target), key=lambda value: counts[value])
        buckets[index].append(sentence)
        counts[index] += len(sentence.split())
    return [" ".join(bucket) for bucket in buckets]


def _read_pcm16(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError("audio must be 16-bit PCM WAV")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    if channels > 1:
        mono = array("h")
        for offset in range(0, len(samples), channels):
            mono.append(int(sum(samples[offset:offset + channels]) / channels))
        samples = mono
    return samples, rate


def analyze_audio(path: Path, narration: str, settings: Settings, target_wpm: int) -> AudioMetrics:
    samples, rate = _read_pcm16(path)
    normalized = [sample / 32768 for sample in samples]
    peak = max(abs(value) for value in normalized)
    rms = math.sqrt(sum(value * value for value in normalized) / max(len(normalized), 1))
    peak_dbfs = 20 * math.log10(max(peak, 1e-12))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    clipping = sum(abs(value) >= 0.999 for value in normalized) / max(len(normalized), 1)
    threshold = 10 ** (-42 / 20)
    window = max(1, int(rate * 0.02))
    silent = []
    for start in range(0, len(normalized), window):
        values = normalized[start:start + window]
        window_rms = math.sqrt(sum(value * value for value in values) / max(len(values), 1))
        silent.append(window_rms < threshold)
    longest = current = 0
    for value in silent:
        current = current + 1 if value else 0
        longest = max(longest, current)
    duration = len(normalized) / rate
    wpm = len(narration.split()) / max(duration, 0.001) * 60
    silence_ratio = sum(silent) / max(len(silent), 1)
    failures = []
    if peak_dbfs > settings.audio_peak_limit_dbfs + 0.05:
        failures.append("peak limit exceeded")
    if rms_dbfs < settings.audio_min_rms_dbfs:
        failures.append("audio is too quiet")
    if clipping > settings.audio_max_clipping_ratio:
        failures.append("clipping ratio exceeded")
    if silence_ratio > settings.audio_max_silence_ratio:
        failures.append("silence ratio exceeded")
    if longest * window / rate > settings.audio_max_silence_seconds:
        failures.append("dead air exceeded")
    if abs(wpm - target_wpm) > settings.audio_wpm_tolerance:
        failures.append("speaking pace outside contract")
    return AudioMetrics(duration, rate, 1, peak_dbfs, rms_dbfs, clipping, silence_ratio, longest * window / rate, wpm, sum(normalized) / max(len(normalized), 1), not failures, tuple(failures))


def concatenate_segments(segments: list[NarrationSegment], output: Path, pause_ms: int) -> tuple[Path, list[NarrationSegment]]:
    ordered = sorted(segments, key=lambda item: item.segment_id)
    all_samples = array("h")
    timed = []
    cursor = 0.0
    rate = None
    for index, segment in enumerate(ordered):
        samples, segment_rate = _read_pcm16(segment.audio_path)
        rate = rate or segment_rate
        if segment_rate != rate:
            raise RuntimeError("inconsistent sample rates")
        start = cursor
        all_samples.extend(samples)
        cursor += len(samples) / rate
        timed.append(replace(segment, start_seconds=start, end_seconds=cursor))
        if index + 1 < len(ordered):
            pause = int(rate * pause_ms / 1000)
            all_samples.extend(array("h", [0]) * pause)
            cursor += pause / rate
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate or 24000)
        wav.writeframes(all_samples.tobytes())
    return output, timed


def normalize_audio(source: Path, output: Path) -> Path:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-af", "loudnorm=I=-16:TP=-1:LRA=11", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode or not output.exists():
        raise RuntimeError(f"audio normalization failed: {result.stderr[-1000:]}")
    return output


def write_manifest(path: Path, segments: list[NarrationSegment], metrics: AudioMetrics, reviews: list[dict], generator: dict, reviewer: dict) -> Path:
    path.write_text(json.dumps({"schema_version": 2, "generator": generator, "reviewer": reviewer, "metrics": metrics.as_dict(), "segments": [{"segment_id": item.segment_id, "text": item.text, "start_seconds": item.start_seconds, "end_seconds": item.end_seconds, "attempt": item.attempt} for item in segments], "reviews": reviews}, indent=2), encoding="utf-8")
    return path
