from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from .config import Settings
from .models import NarrationSegment, VideoPackage
from .policy import Strategy

THEMES = {
    "dashboard": ("#061522", "#0D2B42", "#4AD7FF", "#43E6A0", "#F6FBFF"),
    "kinetic": ("#140A24", "#2B1646", "#FF5DA2", "#FFD166", "#FFF7FC"),
    "cinematic": ("#071014", "#13252A", "#C8FF4A", "#4AC8FF", "#F7FFF1"),
}


def _font(size: int, bold: bool = False):
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _frame(settings: Settings, package: VideoPackage, strategy: Strategy, index: int, path: Path) -> None:
    background, panel, primary, secondary, text = THEMES.get(strategy.visual, THEMES["dashboard"])
    scene = package.scenes[index]
    image = Image.new("RGB", (settings.width, settings.height), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, settings.width, 18), fill=primary)
    draw.rounded_rectangle((55, 110, 230, 165), radius=22, fill=primary)
    draw.text((82, 124), "AI SIGNAL", font=_font(22, True), fill=background)
    draw.multiline_text((55, 215), "\n".join(textwrap.wrap(scene.heading.upper(), 16)), font=_font(54, True), fill=text, spacing=10)
    draw.multiline_text((55, 360), "\n".join(textwrap.wrap(scene.body, 34)), font=_font(29), fill=secondary, spacing=8)
    draw.rounded_rectangle((55, 505, settings.width - 55, 1035), radius=30, fill=panel, outline=primary, width=4)
    draw.text((90, 555), f"STEP {index + 1}", font=_font(28, True), fill=primary)
    draw.multiline_text((90, 635), "\n".join(textwrap.wrap(scene.visual, 30)), font=_font(25), fill=text, spacing=9)
    publisher = package.source_publishers[scene.source_index]
    draw.text((55, settings.height - 175), f"Source: {publisher}", font=_font(20), fill=secondary)
    draw.rectangle((55, settings.height - 105, settings.width - 55, settings.height - 85), fill=panel)
    draw.rectangle((55, settings.height - 105, 55 + int((settings.width - 110) * (index + 1) / 6), settings.height - 85), fill=secondary)
    image.save(path, quality=94)


def _thumbnail(package: VideoPackage, strategy: Strategy, path: Path) -> None:
    background, _, primary, secondary, text = THEMES.get(strategy.visual, THEMES["dashboard"])
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 28, 720), fill=primary)
    draw.text((80, 70), "AI SIGNAL", font=_font(34, True), fill=secondary)
    draw.multiline_text((80, 190), "\n".join(textwrap.wrap(package.thumbnail_text.upper(), 16)), font=_font(102, True), fill=text, spacing=12)
    image.save(path, quality=94)


def _srt(segments: list[NarrationSegment], path: Path) -> None:
    def stamp(seconds: float) -> str:
        ms = int(max(seconds, 0) * 1000)
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, ms = divmod(ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    lines = []
    for index, segment in enumerate(sorted(segments, key=lambda item: item.segment_id), 1):
        lines += [str(index), f"{stamp(segment.start_seconds)} --> {stamp(segment.end_seconds)}", segment.text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_video(settings: Settings, package: VideoPackage, strategy: Strategy, workdir: Path, segments: list[NarrationSegment]) -> tuple[Path, Path]:
    if len(package.scenes) != 6 or len(segments) != 6:
        raise RuntimeError("renderer requires six scenes and six narration segments")
    workdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(6):
        path = workdir / f"scene-{index:02d}.png"
        _frame(settings, package, strategy, index, path)
        frames.append(path)
    thumbnail = workdir / "thumbnail.jpg"
    _thumbnail(package, strategy, thumbnail)
    _srt(segments, workdir / "captions.srt")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    scene_videos = []
    for index, (frame, segment) in enumerate(zip(frames, segments, strict=True)):
        output = workdir / f"scene-{index:02d}.mp4"
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(frame), "-t", f"{max(segment.duration_seconds, 0.25):.4f}", "-vf", f"scale={settings.width}:{settings.height}:flags=lanczos,format=yuv420p", "-r", str(settings.fps), "-c:v", "libx264", "-crf", "18", "-an", str(output)], capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        scene_videos.append(output)
    concat = workdir / "scenes.txt"
    concat.write_text("\n".join(f"file '{path.as_posix()}'" for path in scene_videos), encoding="utf-8")
    silent = workdir / "silent.mp4"
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(silent)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    final = workdir / "video.mp4"
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(workdir / "voice.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(final)], capture_output=True, text=True, timeout=300)
    if result.returncode or not final.exists():
        raise RuntimeError(f"final render failed: {result.stderr[-1000:]}")
    return final, thumbnail
