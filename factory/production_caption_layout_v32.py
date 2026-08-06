from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import imageio_ffmpeg
from PIL import Image, ImageChops, ImageFont

from .models import NarrationSegment


_INSTALLED = False


def _ass_filter_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


def _font(font_name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        font_name,
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line_width(text: str, *, font_name: str, font_size: int, outline: int) -> int:
    font = _font(font_name, font_size)
    left, _top, right, _bottom = font.getbbox(text, stroke_width=outline)
    return max(0, int(right - left))


def fit_caption_lines(
    text: str,
    *,
    width: int,
    height: int,
    font_name: str = "DejaVu Sans",
    horizontal_margin: int | None = None,
    outline: int = 3,
) -> dict[str, Any]:
    """Fit one caption into one or two centered lines without touching the frame edges."""
    words = text.split()
    if not words:
        raise ValueError("Caption text is empty")
    margin = horizontal_margin or max(84, int(round(width * 0.09)))
    safe_width = width - (2 * margin) - (2 * outline)
    if safe_width <= 0:
        raise ValueError("Caption horizontal safe area is invalid")
    base_size = max(42, int(round(height * 0.035)))
    minimum_size = max(42, int(round(base_size * 0.76)))

    for font_size in range(base_size, minimum_size - 1, -1):
        full_width = _line_width(
            text,
            font_name=font_name,
            font_size=font_size,
            outline=outline,
        )
        if full_width <= safe_width:
            return {
                "lines": [text],
                "font_size": font_size,
                "maximum_line_width_pixels": full_width,
                "safe_width_pixels": safe_width,
                "horizontal_margin_pixels": margin,
            }

        best: tuple[int, list[str]] | None = None
        for split in range(1, len(words)):
            lines = [" ".join(words[:split]), " ".join(words[split:])]
            widths = [
                _line_width(
                    line,
                    font_name=font_name,
                    font_size=font_size,
                    outline=outline,
                )
                for line in lines
            ]
            maximum = max(widths)
            if maximum <= safe_width and (best is None or maximum < best[0]):
                best = (maximum, lines)
        if best is not None:
            return {
                "lines": best[1],
                "font_size": font_size,
                "maximum_line_width_pixels": best[0],
                "safe_width_pixels": safe_width,
                "horizontal_margin_pixels": margin,
            }

    raise ValueError(
        f"Caption cannot fit inside {safe_width}px using at most two lines: {text!r}"
    )


def _rendered_caption_bounds(
    caption_path: Path,
    cues: Sequence[Any],
    *,
    width: int,
    height: int,
    horizontal_margin: int,
    outline: int,
) -> list[tuple[int, int, int, int]]:
    """Render every cue through libass and prove the actual pixels stay platform-safe."""
    if not cues:
        return []
    fps = 30
    frame_numbers = [
        max(0, round(((float(cue.start_seconds) + float(cue.end_seconds)) / 2.0) * fps))
        for cue in cues
    ]
    selection = "+".join(f"eq(n\\,{number})" for number in frame_numbers)
    duration = max(float(cue.end_seconds) for cue in cues) + 0.25
    with tempfile.TemporaryDirectory(prefix="caption-proof-") as temporary:
        output_pattern = Path(temporary) / "proof-%04d.png"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}:d={duration:.3f}",
            "-vf",
            (
                f"subtitles=filename='{_ass_filter_path(caption_path)}':"
                "fontsdir='/usr/share/fonts/truetype/dejavu',"
                f"select='{selection}'"
            ),
            "-vsync",
            "0",
            str(output_pattern),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        frames = sorted(Path(temporary).glob("proof-*.png"))
        if completed.returncode != 0 or len(frames) != len(cues):
            raise RuntimeError(
                "Rendered caption proof failed: "
                + (completed.stderr[-2000:] or f"expected {len(cues)} frames, got {len(frames)}")
            )

        minimum_x = max(0, horizontal_margin - outline - 8)
        maximum_x = min(width, width - horizontal_margin + outline + 8)
        minimum_y = round(height * 0.66)
        maximum_y = round(height * 0.86)
        bounds: list[tuple[int, int, int, int]] = []
        black = Image.new("RGB", (width, height), "black")
        for cue, frame_path in zip(cues, frames, strict=True):
            with Image.open(frame_path) as frame:
                bbox = ImageChops.difference(frame.convert("RGB"), black).getbbox()
            if bbox is None:
                raise RuntimeError(f"Caption cue {cue.cue_id} rendered no visible pixels")
            left, top, right, bottom = bbox
            if left < minimum_x or right > maximum_x or top < minimum_y or bottom > maximum_y:
                raise RuntimeError(
                    f"Caption cue {cue.cue_id} rendered outside the safe region: "
                    f"bbox={bbox}, safe=({minimum_x}, {minimum_y}, {maximum_x}, {maximum_y})"
                )
            bounds.append((left, top, right, bottom))
        return bounds


def write_pixel_fitted_caption_track(
    segments: Sequence[NarrationSegment],
    output_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "DejaVu Sans",
) -> tuple[Any, ...]:
    """Write a separate ASS layer with deterministic per-cue pixel fitting."""
    from . import caption_renderer

    cues = caption_renderer.build_caption_cues(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    margin_v = int(round(height * 0.21))
    center_x = width // 2
    baseline_y = height - margin_v
    base_size = max(42, int(round(height * 0.035)))
    horizontal_margin = max(84, int(round(width * 0.09)))
    outline = 3

    layouts: list[dict[str, Any]] = []
    for cue in cues:
        layout = fit_caption_lines(
            cue.text,
            width=width,
            height=height,
            font_name=font_name,
            horizontal_margin=horizontal_margin,
            outline=outline,
        )
        layouts.append(layout)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{base_size},&H00FFFFFF,&H0000D7FF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,3,{outline},0,2,{horizontal_margin},{horizontal_margin},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    manifest_cues: list[dict[str, Any]] = []
    for cue, layout in zip(cues, layouts, strict=True):
        rendered = r"\N".join(
            caption_renderer._escape_ass(line) for line in layout["lines"]
        )
        animation = (
            rf"{{\an2\pos({center_x},{baseline_y})\fs{layout['font_size']}"
            rf"\fad(70,90)\fscx92\fscy92\t(0,140,\fscx100\fscy100)}}"
        )
        lines.append(
            "Dialogue: 0,"
            + caption_renderer._ass_time(cue.start_seconds)
            + ","
            + caption_renderer._ass_time(cue.end_seconds)
            + ",Caption,,0,0,0,,"
            + animation
            + rendered
            + "\n"
        )
        manifest_cues.append({**cue.as_dict(), "layout": layout})

    output_path.write_text("".join(lines), encoding="utf-8")
    rendered_bounds = _rendered_caption_bounds(
        output_path,
        cues,
        width=width,
        height=height,
        horizontal_margin=horizontal_margin,
        outline=outline,
    )
    for item, bounds in zip(manifest_cues, rendered_bounds, strict=True):
        item["rendered_bbox_pixels"] = list(bounds)
    maximum_width = max(
        int(layout["maximum_line_width_pixels"]) for layout in layouts
    ) if layouts else 0
    safe_width = min(int(layout["safe_width_pixels"]) for layout in layouts) if layouts else 0
    manifest = {
        "format": "ass",
        "rendering": "separate_animated_caption_layer",
        "layout_version": "v32-pixel-fitted-two-line",
        "width": width,
        "height": height,
        "font_name": font_name,
        "font_size": base_size,
        "safe_zone": "bottom_21_percent_baseline; platform controls remain below captions",
        "horizontal_safe_margin_pixels": horizontal_margin,
        "safe_line_width_pixels": safe_width,
        "maximum_rendered_line_width_pixels": maximum_width,
        "all_cues_fit": bool(layouts) and maximum_width <= safe_width,
        "all_rendered_cues_fit": bool(rendered_bounds) and len(rendered_bounds) == len(cues),
        "cues": manifest_cues,
    }
    if not manifest["all_cues_fit"] or not manifest["all_rendered_cues_fit"]:
        raise caption_renderer.CaptionRenderError("Caption pixel-fit verification failed")
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cues


def install_production_caption_layout_v32() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import caption_renderer

    caption_renderer.write_animated_caption_track = write_pixel_fitted_caption_track
    _INSTALLED = True
