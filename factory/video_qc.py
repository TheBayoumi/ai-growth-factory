from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageStat

from .config import Settings


class VideoQCError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoQCReport:
    passed: bool
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    file_size_bytes: int
    frame_luminance: tuple[float, ...]
    frame_contrast: tuple[float, ...]
    frame_differences: tuple[float, ...]
    temporal_window_mean_differences: tuple[float, ...]
    temporal_window_near_static_ratios: tuple[float, ...]
    temporal_window_jump_ratios: tuple[float, ...]
    temporal_window_max_differences: tuple[float, ...]
    temporal_stutter_windows: int
    thumbnail_width: int
    thumbnail_height: int
    thumbnail_contrast: float
    production_voice_verified: bool
    voice_provenance: str
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / max(float(denominator), 1.0)
    return float(value)


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoQCError("ffprobe is required for output verification")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise VideoQCError(f"ffprobe failed: {completed.stderr[-2000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoQCError("ffprobe returned invalid JSON") from exc


def _extract_frames(video: Path, duration: float, output_dir: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoQCError("ffmpeg is required for output verification")
    points = [max(0.05, duration * fraction) for fraction in (0.08, 0.33, 0.58, 0.88)]
    frames: list[Path] = []
    for index, timestamp in enumerate(points):
        output = output_dir / f"sample-{index}.png"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode != 0 or not output.exists():
            raise VideoQCError(f"Frame extraction failed: {completed.stderr[-2000:]}")
        frames.append(output)
    return frames


def _frame_stats(paths: list[Path]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    luminance: list[float] = []
    contrast: list[float] = []
    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("L").resize((180, 320))
        images.append(image)
        stats = ImageStat.Stat(image)
        luminance.append(float(stats.mean[0]))
        contrast.append(float(stats.stddev[0]))
    differences: list[float] = []
    for first, second in zip(images, images[1:]):
        diff = ImageChops.difference(first, second)
        differences.append(float(ImageStat.Stat(diff).mean[0]))
    return tuple(luminance), tuple(contrast), tuple(differences)


def _scene_centers(duration: float, scene_durations: Sequence[float] | None) -> list[float]:
    if scene_durations:
        clean = [max(0.1, float(item)) for item in scene_durations]
        total = sum(clean)
        if total > 0:
            scale = duration / total
            cursor = 0.0
            centers: list[float] = []
            for item in clean:
                adjusted = item * scale
                centers.append(cursor + adjusted / 2.0)
                cursor += adjusted
            return centers
    return [duration * (index + 0.5) / 6.0 for index in range(6)]


def _raw_gray_frames(video: Path, *, center: float, sample_seconds: float = 1.0) -> list[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoQCError("ffmpeg is required for temporal verification")
    width, height, sample_fps = 90, 160, 12
    start = max(0.0, center - sample_seconds / 2.0)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video),
        "-t",
        f"{sample_seconds:.3f}",
        "-vf",
        f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise VideoQCError(f"Temporal frame extraction failed: {stderr[-2000:]}")
    frame_size = width * height
    payload = completed.stdout
    return [
        payload[index : index + frame_size]
        for index in range(0, len(payload) - frame_size + 1, frame_size)
    ]


def _mean_abs_difference(first: bytes, second: bytes) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)



def _is_hold_jump_stutter(differences: Sequence[float]) -> bool:
    if not differences:
        return False
    near_static = sum(value < 0.10 for value in differences) / len(differences)
    jumps = sum(value > 0.90 for value in differences) / len(differences)
    maximum = max(differences)
    return near_static >= 0.25 and jumps >= 0.12 and maximum >= 1.8

def _temporal_stability(
    video: Path,
    duration: float,
    scene_durations: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
    means: list[float] = []
    near_static_ratios: list[float] = []
    jump_ratios: list[float] = []
    maxima: list[float] = []
    stutter_windows = 0
    for center in _scene_centers(duration, scene_durations):
        frames = _raw_gray_frames(video, center=center)
        differences = [
            _mean_abs_difference(first, second)
            for first, second in zip(frames, frames[1:])
        ]
        if not differences:
            means.append(0.0)
            near_static_ratios.append(1.0)
            jump_ratios.append(0.0)
            maxima.append(0.0)
            continue
        window_mean = mean(differences)
        near_static = sum(value < 0.10 for value in differences) / len(differences)
        jumps = sum(value > 0.90 for value in differences) / len(differences)
        maximum = max(differences)
        stutter = _is_hold_jump_stutter(differences)
        if stutter:
            stutter_windows += 1
        means.append(window_mean)
        near_static_ratios.append(near_static)
        jump_ratios.append(jumps)
        maxima.append(maximum)
    return (
        tuple(means),
        tuple(near_static_ratios),
        tuple(jump_ratios),
        tuple(maxima),
        stutter_windows,
    )


def _verify_voice_manifest(
    settings: Settings,
    path: Path | None,
) -> tuple[bool, str, list[str]]:
    failures: list[str] = []
    if path is None or not path.exists():
        return False, "missing", ["production voice manifest is missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, "invalid", [f"production voice manifest is invalid: {exc}"]
    generator = data.get("generator") or {}
    backend = str(generator.get("backend") or "").strip().lower()
    model = str(generator.get("model") or "").strip()
    provenance = f"{backend or 'unknown'}:{model or 'unknown'}"
    if backend != "qwen3" or "qwen3-tts" not in model.lower():
        failures.append(f"voice generator provenance is not approved Qwen3-TTS: {provenance}")
    metrics = data.get("metrics") or {}
    if metrics.get("passed") is not True:
        failures.append("voice deterministic QC did not pass")
    if settings.reviewer_required:
        reviews = [
            item
            for item in data.get("reviews") or []
            if isinstance(item, dict) and item.get("type") == "model_review"
        ]
        if not reviews:
            failures.append("required perceptual voice review is missing")
        else:
            final_review = reviews[-1]
            if final_review.get("decision") != "approve":
                failures.append("final perceptual voice review was not approved")
            if float(final_review.get("overall_score") or 0.0) < settings.reviewer_overall_threshold:
                failures.append("final perceptual voice score is below the local threshold")
            scores = final_review.get("scores") or {}
            if float(scores.get("naturalness") or 0.0) < settings.reviewer_naturalness_threshold:
                failures.append("voice naturalness is below the local threshold")
            if float(scores.get("pronunciation") or 0.0) < settings.reviewer_pronunciation_threshold:
                failures.append("voice pronunciation is below the local threshold")
            if float(scores.get("script_fidelity") or 0.0) < settings.reviewer_fidelity_threshold:
                failures.append("voice script fidelity is below the local threshold")
    return not failures, provenance, failures


def verify_video_output(
    settings: Settings,
    video_path: Path,
    thumbnail_path: Path,
    *,
    expected_duration: float | None = None,
    scene_durations: Sequence[float] | None = None,
    voice_manifest_path: Path | None = None,
    require_production_voice: bool = False,
    report_path: Path | None = None,
) -> VideoQCReport:
    if not video_path.exists():
        raise VideoQCError(f"Video does not exist: {video_path}")
    if not thumbnail_path.exists():
        raise VideoQCError(f"Thumbnail does not exist: {thumbnail_path}")

    probe = _probe(video_path)
    streams = probe.get("streams") or []
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video_stream or not audio_stream:
        raise VideoQCError("Rendered file must contain video and audio streams")

    duration = float((probe.get("format") or {}).get("duration") or 0.0)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_rate(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0"))
    video_codec = str(video_stream.get("codec_name") or "")
    audio_codec = str(audio_stream.get("codec_name") or "")
    audio_sample_rate = int(audio_stream.get("sample_rate") or 0)
    audio_channels = int(audio_stream.get("channels") or 0)

    with tempfile.TemporaryDirectory(prefix="video-qc-") as temporary:
        frames = _extract_frames(video_path, duration, Path(temporary))
        luminance, contrast, differences = _frame_stats(frames)

    temporal = _temporal_stability(video_path, duration, scene_durations)
    temporal_means, temporal_near_static, temporal_jumps, temporal_maxima, stutter_windows = temporal

    thumbnail = Image.open(thumbnail_path).convert("L")
    thumbnail_width, thumbnail_height = thumbnail.size
    thumbnail_contrast = float(ImageStat.Stat(thumbnail.resize((320, 180))).stddev[0])

    failures: list[str] = []
    if width != settings.width or height != settings.height:
        failures.append(
            f"resolution {width}x{height} does not match {settings.width}x{settings.height}"
        )
    if width >= height:
        failures.append("video is not portrait-oriented")
    if video_codec != "h264":
        failures.append(f"video codec is {video_codec or 'missing'}, expected h264")
    if audio_codec != "aac":
        failures.append(f"audio codec is {audio_codec or 'missing'}, expected aac")
    if audio_channels not in {1, 2}:
        failures.append(f"audio channel count {audio_channels} is invalid")
    if audio_sample_rate < 44100:
        failures.append(f"audio sample rate {audio_sample_rate} is below 44.1 kHz")
    if abs(fps - settings.fps) > 0.6:
        failures.append(f"frame rate {fps:.3f} does not match configured {settings.fps}")
    if duration < 15.0:
        failures.append(f"duration {duration:.2f}s is too short for a publication canary")
    if duration > 100.0:
        failures.append(f"duration {duration:.2f}s exceeds the short-form ceiling")
    if expected_duration is not None and abs(duration - expected_duration) > 1.25:
        failures.append(
            f"video duration {duration:.2f}s differs from narration {expected_duration:.2f}s"
        )
    if video_path.stat().st_size < 250_000:
        failures.append("video file is unexpectedly small")
    if any(value < 16.0 for value in luminance):
        failures.append("one or more sampled frames are excessively dark")
    if any(value < 18.0 for value in contrast):
        failures.append("one or more sampled frames have insufficient visual contrast")
    if differences and mean(differences) < 4.0:
        failures.append("sampled scenes are not visually distinct enough")
    if stutter_windows:
        failures.append(
            f"temporal stability failed in {stutter_windows} scene window(s): hold-jump motion detected"
        )
    if thumbnail_width != 1280 or thumbnail_height != 720:
        failures.append(
            f"thumbnail resolution {thumbnail_width}x{thumbnail_height} is not 1280x720"
        )
    if thumbnail_contrast < 22.0:
        failures.append("thumbnail contrast is too low")

    production_voice_verified = False
    voice_provenance = "not-required"
    if require_production_voice:
        production_voice_verified, voice_provenance, voice_failures = _verify_voice_manifest(
            settings, voice_manifest_path
        )
        failures.extend(voice_failures)

    report = VideoQCReport(
        passed=not failures,
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        file_size_bytes=video_path.stat().st_size,
        frame_luminance=luminance,
        frame_contrast=contrast,
        frame_differences=differences,
        temporal_window_mean_differences=temporal_means,
        temporal_window_near_static_ratios=temporal_near_static,
        temporal_window_jump_ratios=temporal_jumps,
        temporal_window_max_differences=temporal_maxima,
        temporal_stutter_windows=stutter_windows,
        thumbnail_width=thumbnail_width,
        thumbnail_height=thumbnail_height,
        thumbnail_contrast=thumbnail_contrast,
        production_voice_verified=production_voice_verified,
        voice_provenance=voice_provenance,
        failures=tuple(failures),
    )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    if not report.passed:
        raise VideoQCError("; ".join(report.failures))
    return report
