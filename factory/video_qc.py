from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageStat

from .config import Settings


@dataclass(frozen=True)
class VideoQCResult:
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int
    thumbnail_contrast: float
    passed: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _ffprobe() -> str:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    candidate = ffmpeg.with_name("ffprobe.exe" if ffmpeg.name.endswith(".exe") else "ffprobe")
    return str(candidate if candidate.exists() else shutil.which("ffprobe") or "ffprobe")


def _manifest_failures(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    generator = data.get("generator") or {}
    if generator.get("backend") != "qwen3" or "qwen3-tts" not in str(generator.get("model", "")).lower():
        failures.append("voice lacks Qwen3-TTS provenance")
    reviews = [item for item in data.get("reviews", []) if item.get("type") == "model_review"]
    if not reviews or reviews[-1].get("decision") != "approve":
        failures.append("voice lacks final perceptual approval")
    return failures


def verify_video_output(settings: Settings, video_path: Path, thumbnail_path: Path, expected_duration: float, voice_manifest_path: Path) -> VideoQCResult:
    result = subprocess.run([_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    data = json.loads(result.stdout)
    video = next((item for item in data.get("streams", []) if item.get("codec_type") == "video"), {})
    audio = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration", 0))
    rate = str(video.get("avg_frame_rate", "0/1")).split("/")
    fps = float(rate[0]) / max(float(rate[1]), 1)
    failures = _manifest_failures(voice_manifest_path)
    if (int(video.get("width", 0)), int(video.get("height", 0))) != (settings.width, settings.height):
        failures.append("video dimensions do not match configuration")
    if abs(duration - expected_duration) > max(0.65, expected_duration * 0.015):
        failures.append("video duration differs from narration")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        failures.append("unexpected codecs")
    if int(audio.get("sample_rate", 0)) != 48000:
        failures.append("audio sample rate must be 48000 Hz")
    with Image.open(thumbnail_path) as image:
        contrast = float(ImageStat.Stat(image.convert("L")).stddev[0])
        if image.size != (1280, 720) or contrast < 20:
            failures.append("thumbnail failed dimension or contrast gate")
    outcome = VideoQCResult(duration, int(video.get("width", 0)), int(video.get("height", 0)), fps, str(video.get("codec_name", "")), str(audio.get("codec_name", "")), int(audio.get("sample_rate", 0)), contrast, not failures, tuple(failures))
    if failures:
        raise RuntimeError("post-render verification failed: " + "; ".join(failures))
    return outcome
