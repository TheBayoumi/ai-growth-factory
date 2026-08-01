from __future__ import annotations

import math
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import Settings
from .models import NarrationSegment, VideoPackage
from .policy import Strategy


BRAND = "AI SIGNAL"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, width: int, *, max_lines: int | None = None) -> list[str]:
    lines = textwrap.wrap(
        " ".join(text.split()),
        width=max(12, width),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    return lines


def _gradient(width: int, height: int, phase: int, visual_style: str) -> Image.Image:
    image = Image.new("RGB", (width, height), "#07111f")
    pixels = image.load()
    style_shift = {"kinetic": 0.11, "dashboard": 0.24, "cinematic": 0.42}.get(
        visual_style, 0.24
    )
    for y in range(height):
        for x in range(width):
            nx = x / max(width - 1, 1)
            ny = y / max(height - 1, 1)
            cx = (style_shift + phase * 0.13) % 0.88
            cy = 0.18 + (phase % 3) * 0.12
            glow = max(0.0, 1.0 - math.hypot(nx - cx, ny - cy))
            edge = max(0.0, 1.0 - math.hypot(nx - 0.88, ny - 0.72))
            pixels[x, y] = (
                int(6 + 21 * glow + 7 * ny + 10 * edge),
                int(16 + 63 * glow + 12 * nx + 18 * edge),
                int(31 + 94 * glow + 15 * (1 - ny) + 25 * edge),
            )
    return image.filter(ImageFilter.GaussianBlur(radius=1.1))


def _text_height(font: ImageFont.ImageFont, spacing: float = 1.2) -> int:
    size = getattr(font, "size", 32)
    return int(size * spacing)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    index: int,
    scene_count: int,
    source: str | None,
) -> None:
    margin = int(width * 0.07)
    draw.text(
        (margin, int(height * 0.045)),
        BRAND,
        font=_font(max(22, width // 28), True),
        fill=(112, 227, 255, 255),
    )
    counter = f"{index + 1:02d}/{scene_count:02d}"
    counter_font = _font(max(20, width // 31), True)
    bbox = draw.textbbox((0, 0), counter, font=counter_font)
    draw.text(
        (width - margin - (bbox[2] - bbox[0]), int(height * 0.045)),
        counter,
        font=counter_font,
        fill=(218, 235, 247, 220),
    )
    if source:
        label = f"SOURCE · {source.upper()}"
        source_font = _font(max(15, width // 42), True)
        bbox = draw.textbbox((0, 0), label, font=source_font)
        pad_x = 13
        x = width - margin - (bbox[2] - bbox[0]) - pad_x * 2
        y = int(height * 0.095)
        draw.rounded_rectangle(
            (x, y, width - margin, y + bbox[3] - bbox[1] + 14),
            radius=12,
            fill=(12, 38, 58, 220),
            outline=(101, 218, 255, 80),
        )
        draw.text((x + pad_x, y + 5), label, font=source_font, fill=(182, 226, 244, 235))


def _draw_progress(
    draw: ImageDraw.ImageDraw, width: int, height: int, index: int, scene_count: int
) -> None:
    margin = int(width * 0.07)
    y = int(height * 0.946)
    draw.rounded_rectangle(
        (margin, y, width - margin, y + max(7, height // 180)),
        radius=5,
        fill=(255, 255, 255, 38),
    )
    progress = margin + (width - 2 * margin) * (index + 1) / scene_count
    draw.rounded_rectangle(
        (margin, y, progress, y + max(7, height // 180)),
        radius=5,
        fill=(91, 222, 255, 235),
    )


def _draw_visual_motif(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    index: int,
    strategy: Strategy,
    visual_hint: str,
) -> None:
    left = int(width * 0.09)
    right = int(width * 0.91)
    top = int(height * 0.17)
    bottom = int(height * 0.47)
    accent = (74, 211, 255, 210)
    soft = (74, 211, 255, 95)
    bright = (221, 248, 255, 245)
    hint = visual_hint.lower()

    if "cost" in hint:
        baseline = bottom
        labels = (("HOSTED", 0.86), ("FREE-FIRST", 0.32))
        bar_width = int((right - left) * 0.23)
        positions = (int(width * 0.28), int(width * 0.63))
        for (label, value), x in zip(labels, positions):
            y = baseline - int((bottom - top) * value)
            draw.rounded_rectangle(
                (x, y, x + bar_width, baseline),
                radius=max(9, width // 75),
                fill=accent if value < 0.5 else (116, 153, 179, 125),
            )
            font = _font(max(16, width // 39), True)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text(
                (x + (bar_width - (bbox[2] - bbox[0])) / 2, baseline + 12),
                label,
                font=font,
                fill=(205, 230, 242, 220),
            )
        draw.line((left, baseline, right, baseline), fill=(179, 224, 242, 95), width=2)
        draw.text((int(width * 0.54), top), "↓", font=_font(max(70, width // 7), True), fill=bright)
        return

    if "source" in hint or "graph" in hint:
        center = (width // 2, int(height * 0.32))
        radius = max(32, width // 13)
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            fill=(14, 54, 76, 245),
            outline=accent,
            width=max(3, width // 180),
        )
        draw.text(
            (center[0] - radius // 2, center[1] - 18),
            "MODEL",
            font=_font(max(17, width // 37), True),
            fill=bright,
        )
        nodes = [
            (int(width * 0.18), int(height * 0.22)),
            (int(width * 0.82), int(height * 0.20)),
            (int(width * 0.16), int(height * 0.42)),
            (int(width * 0.84), int(height * 0.43)),
        ]
        for node_index, (x, y) in enumerate(nodes):
            draw.line((x, y, center[0], center[1]), fill=soft, width=max(2, width // 220))
            r = max(18, width // 26)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(40, 134, 174, 220), outline=bright, width=2)
            draw.text((x - 8, y - 12), str(node_index + 1), font=_font(max(18, width // 35), True), fill=bright)
        return

    if "timeline" in hint or "scene" in hint:
        y = int(height * 0.31)
        draw.line((left, y, right, y), fill=soft, width=max(3, width // 190))
        count = 6
        step = (right - left) / (count - 1)
        for item in range(count):
            x = left + item * step
            r = max(20, width // 24)
            active = item <= index
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=accent if active else (30, 73, 98, 220),
                outline=bright if active else (121, 171, 193, 140),
                width=2,
            )
            draw.text((x - 8, y - 13), str(item + 1), font=_font(max(19, width // 34), True), fill=(4, 24, 35, 255) if active else bright)
        return

    if "audio" in hint or "meter" in hint or "wave" in hint:
        center_y = int(height * 0.32)
        samples = 54
        points = []
        for item in range(samples):
            x = left + (right - left) * item / (samples - 1)
            envelope = math.sin(math.pi * item / (samples - 1)) ** 0.7
            value = math.sin(item * 0.92 + index * 0.7) * envelope
            y = center_y - value * (bottom - top) * 0.42
            points.append((x, y))
        draw.line(points, fill=accent, width=max(3, width // 160), joint="curve")
        draw.line((left, center_y, right, center_y), fill=(185, 230, 244, 55), width=1)
        for x in (left, right):
            draw.rounded_rectangle((x, top, x + 9, bottom), radius=4, fill=(121, 225, 255, 90))
        return

    if "repair" in hint or "loop" in hint:
        center_y = int(height * 0.32)
        box_w = int(width * 0.17)
        box_h = int(height * 0.075)
        gap = int(width * 0.055)
        start_x = (width - (box_w * 3 + gap * 2)) // 2
        labels = ("PASS", "RETRY", "PASS")
        for item, label in enumerate(labels):
            x = start_x + item * (box_w + gap)
            fill = (31, 133, 169, 230) if label == "PASS" else (130, 80, 45, 235)
            draw.rounded_rectangle((x, center_y - box_h // 2, x + box_w, center_y + box_h // 2), radius=16, fill=fill, outline=bright, width=2)
            font = _font(max(18, width // 34), True)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x + (box_w - (bbox[2] - bbox[0])) / 2, center_y - 15), label, font=font, fill=bright)
            if item < 2:
                draw.text((x + box_w + 8, center_y - 28), "→", font=_font(max(42, width // 15), True), fill=accent)
        arc_box = (start_x + box_w, center_y - int(height * 0.12), start_x + box_w * 2 + gap, center_y + int(height * 0.12))
        draw.arc(arc_box, 205, 335, fill=soft, width=max(3, width // 180))
        return

    if "verification" in hint or "check" in hint or "codec" in hint:
        labels = ("VOICE", "VIDEO", "SOURCE", "UPLOAD")
        box_h = int(height * 0.055)
        gap = int(height * 0.018)
        y = top
        font = _font(max(18, width // 34), True)
        for label in labels:
            draw.rounded_rectangle((left, y, right, y + box_h), radius=14, fill=(12, 48, 68, 225), outline=soft, width=2)
            draw.text((left + 20, y + 10), label, font=font, fill=bright)
            draw.text((right - 58, y + 5), "✓", font=_font(max(28, width // 23), True), fill=accent)
            y += box_h + gap
        return

    if strategy.visual == "kinetic":
        center_x = width // 2
        center_y = int(height * 0.32)
        for ring in range(5, 0, -1):
            radius = int(width * (0.06 + ring * 0.045))
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=(84, 220, 255, 25 + ring * 18), width=max(2, width // 240))
    else:
        values = [0.28, 0.55, 0.43, 0.79, 0.66, 0.91]
        gap = (right - left) / len(values)
        for item, value in enumerate(values):
            x0 = left + item * gap + gap * 0.14
            x1 = left + (item + 1) * gap - gap * 0.14
            y0 = bottom - (bottom - top) * value
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=(62, 194, 240, 110 + item * 18))

def _draw_caption_panel(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    caption: str,
    index: int,
) -> None:
    margin = int(width * 0.07)
    panel_top = int(height * 0.705)
    panel_bottom = int(height * 0.915)
    draw.rounded_rectangle(
        (margin, panel_top, width - margin, panel_bottom),
        radius=max(22, width // 24),
        fill=(5, 15, 27, 232),
        outline=(92, 216, 255, 78),
        width=max(2, width // 360),
    )
    font = _font(max(27, width // 23), True)
    lines = _wrap(caption, 29 if width <= 720 else 34, max_lines=4)
    line_h = _text_height(font, 1.25)
    total = line_h * len(lines)
    y = panel_top + max(18, (panel_bottom - panel_top - total) // 2)
    accent_words = 4 if index == 0 else 2
    for line_index, line in enumerate(lines):
        fill = (247, 251, 255, 255)
        if line_index == 0 and accent_words:
            words = line.split()
            accent = " ".join(words[:accent_words])
            remainder = " ".join(words[accent_words:])
            x = margin + int(width * 0.045)
            draw.text((x, y), accent, font=font, fill=(101, 226, 255, 255))
            if remainder:
                accent_box = draw.textbbox((x, y), accent + " ", font=font)
                draw.text((accent_box[2], y), remainder, font=font, fill=fill)
        else:
            draw.text((margin + int(width * 0.045), y), line, font=font, fill=fill)
        y += line_h


def _scene_card(
    settings: Settings,
    package: VideoPackage,
    strategy: Strategy,
    index: int,
    caption: str,
    output: Path,
) -> None:
    width, height = settings.width, settings.height
    scene = package.scenes[index]
    image = _gradient(width, height, index, strategy.visual)
    draw = ImageDraw.Draw(image, "RGBA")
    source = None
    if package.source_publishers:
        source_index = min(max(scene.source_index, 0), len(package.source_publishers) - 1)
        source = package.source_publishers[source_index]
    _draw_header(draw, width, height, index, len(package.scenes), source)
    _draw_visual_motif(draw, width, height, index, strategy, scene.visual)

    margin = int(width * 0.075)
    heading_font = _font(max(42, width // 11), True)
    body_font = _font(max(25, width // 25), False)
    heading_lines = _wrap(scene.heading.upper(), 15, max_lines=3)
    y = int(height * 0.50)
    for line in heading_lines:
        draw.text(
            (margin, y),
            line,
            font=heading_font,
            fill=(249, 252, 255, 255),
            stroke_width=max(1, width // 420),
            stroke_fill=(7, 25, 40, 230),
        )
        y += _text_height(heading_font, 1.04)
    body_lines = _wrap(scene.body, 39, max_lines=2)
    y += max(12, height // 80)
    for line in body_lines:
        draw.text((margin, y), line, font=body_font, fill=(192, 219, 237, 245))
        y += _text_height(body_font, 1.28)

    _draw_caption_panel(draw, width, height, caption, index)
    _draw_progress(draw, width, height, index, len(package.scenes))
    image.save(output, optimize=True)


def _thumbnail(settings: Settings, package: VideoPackage, strategy: Strategy, output: Path) -> None:
    width, height = 1280, 720
    image = _gradient(width, height, 2, strategy.visual)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (54, 50, 1226, 670),
        radius=46,
        fill=(5, 15, 28, 224),
        outline=(78, 214, 255, 125),
        width=3,
    )
    draw.text((88, 82), BRAND, font=_font(38, True), fill=(94, 225, 255, 255))
    pill = strategy.hook.replace("-", " ").upper()
    pill_font = _font(25, True)
    pill_box = draw.textbbox((0, 0), pill, font=pill_font)
    draw.rounded_rectangle(
        (90, 145, 125 + pill_box[2], 190),
        radius=17,
        fill=(20, 74, 99, 235),
    )
    draw.text((107, 151), pill, font=pill_font, fill=(178, 237, 255, 255))
    font = _font(96, True)
    lines = _wrap(package.thumbnail_text.upper(), 15, max_lines=3)
    line_h = int(font.size * 1.02) if hasattr(font, "size") else 100
    total = line_h * len(lines)
    y = 230 + max(0, (280 - total) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) / 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(250, 252, 255, 255),
            stroke_width=3,
            stroke_fill=(15, 52, 72, 255),
        )
        y += line_h
    draw.text(
        (90, 618),
        "WHAT CHANGED  •  WHY IT MATTERS",
        font=_font(27, True),
        fill=(201, 224, 240, 235),
    )
    image.save(output, quality=94, optimize=True)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def _scene_durations(
    *,
    audio_duration: float,
    scene_count: int,
    segments: Sequence[NarrationSegment] | None,
) -> list[float]:
    if segments and len(segments) == scene_count:
        ordered = sorted(segments, key=lambda item: item.segment_id)
        durations: list[float] = []
        for index, segment in enumerate(ordered):
            end = ordered[index + 1].start_seconds if index + 1 < len(ordered) else audio_duration
            durations.append(max(0.35, end - segment.start_seconds))
        scale = audio_duration / max(sum(durations), 0.001)
        return [duration * scale for duration in durations]
    return [audio_duration / scene_count for _ in range(scene_count)]


def render_video(
    settings: Settings,
    package: VideoPackage,
    strategy: Strategy,
    workdir: Path,
    *,
    segments: Sequence[NarrationSegment] | None = None,
) -> tuple[Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    if len(package.scenes) != 6:
        raise ValueError("Publication renderer requires exactly six scenes")
    captions = [segment.text for segment in sorted(segments, key=lambda item: item.segment_id)] if segments else []
    if len(captions) != len(package.scenes):
        captions = [scene.body for scene in package.scenes]

    images: list[Path] = []
    for index in range(len(package.scenes)):
        path = workdir / f"scene-{index}.png"
        _scene_card(settings, package, strategy, index, captions[index], path)
        images.append(path)
    thumbnail = workdir / "thumbnail.jpg"
    _thumbnail(settings, package, strategy, thumbnail)

    audio = workdir / "voice.wav"
    if not audio.exists():
        raise FileNotFoundError("voice.wav must exist before rendering")
    duration = _wav_duration(audio)
    if duration < 1.0:
        raise ValueError("Narration is too short to render")
    durations = _scene_durations(
        audio_duration=duration,
        scene_count=len(images),
        segments=segments,
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for image, scene_duration in zip(images, durations):
        command += ["-loop", "1", "-t", f"{scene_duration:.3f}", "-i", str(image)]
    command += ["-i", str(audio)]

    # Text-heavy editorial cards must remain pixel-stable. The previous zoompan
    # implementation quantized crop coordinates to whole pixels, producing long
    # runs of duplicate frames followed by visible one-pixel jumps. Static cards
    # with semantic cuts are intentionally preferred until a true sub-pixel motion
    # renderer is introduced and perceptually validated.
    filters: list[str] = []
    for index, scene_duration in enumerate(durations):
        filters.append(
            f"[{index}:v]scale={settings.width}:{settings.height}:flags=lanczos,"
            f"trim=duration={scene_duration:.3f},setpts=PTS-STARTPTS,"
            f"fps={settings.fps},setsar=1,format=yuv420p[v{index}]"
        )
    filters.append(
        "".join(f"[v{i}]" for i in range(len(images)))
        + f"concat=n={len(images)}:v=1:a=0,fps={settings.fps}[v]"
    )
    output = workdir / "video.mp4"
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        f"{len(images)}:a",
        "-r",
        str(settings.fps),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if completed.returncode != 0 or not output.exists() or output.stat().st_size < 100_000:
        raise RuntimeError(f"FFmpeg render failed: {completed.stderr[-3000:]}")
    return output, thumbnail
