from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .caption_renderer import write_animated_caption_track
from .models import NarrationSegment, VideoPackage
from .video_generator import SceneMediaAsset


class VisualCompositionError(RuntimeError):
    pass


_TRANSITION_SECONDS = 0.10
_THUMBNAIL_WIDTH = 1280
_THUMBNAIL_HEIGHT = 720


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / max(1, handle.getframerate())


def _scene_durations(
    segments: Sequence[NarrationSegment],
    total_duration: float,
) -> list[float]:
    ordered = sorted(segments, key=lambda segment: segment.segment_id)
    durations: list[float] = []
    for index, segment in enumerate(ordered):
        end = (
            ordered[index + 1].start_seconds
            if index + 1 < len(ordered)
            else total_duration
        )
        duration = max(0.5, float(end) - float(segment.start_seconds))
        durations.append(duration)
    scale = total_duration / max(0.001, sum(durations))
    return [duration * scale for duration in durations]


def _filter_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    maximum_size: int,
) -> ImageFont.ImageFont:
    size = maximum_size
    while size >= 32:
        font = _font(size)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=10, align="left")
        if box[2] - box[0] <= width:
            return font
        size -= 4
    return _font(32)


def _wrap_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    font: ImageFont.ImageFont,
    width: int,
) -> str:
    words = title.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if current and box[2] - box[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines[:4])


def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = max(width / image.width, height / image.height)
    resized = image.resize(
        (int(round(image.width * ratio)), int(round(image.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _thumbnail(
    keyframe_path: Path,
    package: VideoPackage,
    output: Path,
    *,
    width: int = _THUMBNAIL_WIDTH,
    height: int = _THUMBNAIL_HEIGHT,
) -> None:
    """Create a dedicated horizontal YouTube thumbnail from a portrait keyframe."""
    with Image.open(keyframe_path) as source:
        portrait = source.convert("RGB")

    background = _cover(portrait, width, height).filter(ImageFilter.GaussianBlur(radius=26))
    background = ImageEnhance.Brightness(background).enhance(0.34)
    background = ImageEnhance.Contrast(background).enhance(1.15)
    canvas = background.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    hero_height = int(height * 0.90)
    hero_width = int(hero_height * portrait.width / max(1, portrait.height))
    hero = portrait.resize((hero_width, hero_height), Image.Resampling.LANCZOS)
    hero = ImageEnhance.Contrast(hero).enhance(1.08)
    hero_x = 54
    hero_y = (height - hero_height) // 2
    shadow = Image.new("RGBA", (hero_width + 28, hero_height + 28), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (14, 14, hero_width + 14, hero_height + 14),
        radius=28,
        fill=(0, 0, 0, 175),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
    canvas.alpha_composite(shadow, (hero_x - 14, hero_y - 14))
    hero_mask = Image.new("L", hero.size, 0)
    ImageDraw.Draw(hero_mask).rounded_rectangle(
        (0, 0, hero_width, hero_height), radius=24, fill=255
    )
    canvas.paste(hero, (hero_x, hero_y), hero_mask)

    panel_left = max(int(width * 0.52), hero_x + hero_width + 44)
    panel_right = width - 70
    panel_width = max(360, panel_right - panel_left)
    draw.rounded_rectangle(
        (panel_left - 28, 82, panel_right + 18, height - 82),
        radius=34,
        fill=(5, 10, 22, 188),
        outline=(85, 200, 255, 150),
        width=3,
    )
    draw.rounded_rectangle(
        (panel_left, 126, panel_left + 116, 138),
        radius=6,
        fill=(82, 210, 255, 255),
    )
    title = package.thumbnail_text.strip().upper() or package.title.strip().upper()
    font = _fit_text(draw, title, panel_width - 18, int(height * 0.105))
    wrapped = _wrap_title(draw, title, font, panel_width - 18)
    box = draw.multiline_textbbox(
        (0, 0), wrapped, font=font, spacing=12, align="left", stroke_width=3
    )
    text_height = box[3] - box[1]
    text_y = max(168, (height - text_height) // 2)
    draw.multiline_text(
        (panel_left, text_y),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=3,
        stroke_fill=(2, 5, 12, 245),
        spacing=12,
        align="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, format="PNG", optimize=True)


def compose_platform_video(
    *,
    media: Sequence[SceneMediaAsset],
    segments: Sequence[NarrationSegment],
    package: VideoPackage,
    audio_path: Path,
    workdir: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> tuple[Path, Path, Path]:
    """Compose text-free scene media and a separate animated caption track."""
    ordered_media = sorted(media, key=lambda asset: asset.scene_index)
    ordered_segments = sorted(segments, key=lambda segment: segment.segment_id)
    if len(ordered_media) != len(package.scenes) or len(ordered_segments) != len(package.scenes):
        raise VisualCompositionError("Media, narration segments, and package scenes must align")
    if [asset.scene_index for asset in ordered_media] != list(range(len(package.scenes))):
        raise VisualCompositionError("Scene media indices are not contiguous")
    if not audio_path.is_file():
        raise VisualCompositionError("Reviewed narration audio is missing")

    workdir.mkdir(parents=True, exist_ok=True)
    caption_path = workdir / "animated-captions.ass"
    cues = write_animated_caption_track(
        ordered_segments,
        caption_path,
        width=width,
        height=height,
    )
    total_duration = _audio_duration(audio_path)
    durations = _scene_durations(ordered_segments, total_duration)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for asset, duration in zip(ordered_media, durations, strict=True):
        if asset.media_type == "image":
            command += [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{duration:.6f}",
                "-i",
                str(asset.path),
            ]
        elif asset.media_type == "video":
            command += [
                "-stream_loop",
                "-1",
                "-t",
                f"{duration:.6f}",
                "-i",
                str(asset.path),
            ]
        else:
            raise VisualCompositionError(f"Unsupported media type: {asset.media_type}")
    audio_index = len(ordered_media)
    command += ["-i", str(audio_path)]

    filters: list[str] = []
    for index, duration in enumerate(durations):
        fade = min(_TRANSITION_SECONDS, max(0.04, duration / 8.0))
        fade_out_start = max(0.0, duration - fade)
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},trim=duration={duration:.6f},"
            f"settb=expr=1/{fps},setpts=N/({fps}*TB),setsar=1,format=yuv420p,"
            f"fade=t=in:st=0:d={fade:.3f},"
            f"fade=t=out:st={fade_out_start:.6f}:d={fade:.3f}[v{index}]"
        )
    filters.append(
        "".join(f"[v{index}]" for index in range(len(ordered_media)))
        + f"concat=n={len(ordered_media)}:v=1:a=0[vcat]"
    )
    subtitles = _filter_path(caption_path)
    filters.append(
        f"[vcat]subtitles=filename='{subtitles}':"
        "fontsdir='/usr/share/fonts/truetype/dejavu'[vout]"
    )

    output = workdir / "video.mp4"
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-map",
        f"{audio_index}:a",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (workdir / "visual-compositor.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 500_000:
        raise VisualCompositionError(
            f"Final visual composition failed: {completed.stderr[-3000:]}"
        )

    thumbnail = workdir / "thumbnail.png"
    _thumbnail(
        ordered_media[0].keyframe_path,
        package,
        thumbnail,
        width=_THUMBNAIL_WIDTH,
        height=_THUMBNAIL_HEIGHT,
    )
    manifest = {
        "renderer": "ffmpeg_ass_hybrid_v2",
        "captions_baked_into_generated_media": False,
        "caption_layer": str(caption_path),
        "caption_cues": len(cues),
        "scene_media": [asset.as_dict() for asset in ordered_media],
        "scene_durations": [round(value, 3) for value in durations],
        "output": {
            "path": str(output),
            "width": width,
            "height": height,
            "fps": fps,
            "audio_path": str(audio_path),
        },
        "thumbnail": {
            "path": str(thumbnail),
            "width": _THUMBNAIL_WIDTH,
            "height": _THUMBNAIL_HEIGHT,
            "profile": "youtube_horizontal",
        },
    }
    (workdir / "visual-composition-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output, thumbnail, caption_path
