from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


_INSTALLED = False


def proportional_caption_margin_v67(width: int) -> int:
    if width <= 0:
        raise ValueError("caption width must be positive")
    return max(48, int(round(width * 0.09)))


def fit_caption_lines_v67(
    text: str,
    *,
    width: int,
    height: int,
    font_name: str = "DejaVu Sans",
    horizontal_margin: int | None = None,
    outline: int = 3,
) -> dict[str, Any]:
    """Scale the safe margin with render width so 720p preflight matches 1080p geometry."""
    from .production_caption_layout_v32 import _line_width

    words = text.split()
    if not words:
        raise ValueError("Caption text is empty")
    margin = horizontal_margin if horizontal_margin is not None else proportional_caption_margin_v67(width)
    safe_width = width - (2 * margin) - (2 * outline)
    if safe_width <= 0:
        raise ValueError("Caption horizontal safe area is invalid")
    base_size = max(28, int(round(height * 0.035)))
    minimum_size = max(28, int(round(base_size * 0.76)))

    for font_size in range(base_size, minimum_size - 1, -1):
        full_width = _line_width(text, font_name=font_name, font_size=font_size, outline=outline)
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
                _line_width(line, font_name=font_name, font_size=font_size, outline=outline)
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
    raise ValueError(f"Caption cannot fit inside {safe_width}px using at most two lines: {text!r}")


def write_scaled_caption_track_v67(
    segments: Sequence[Any],
    output_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "DejaVu Sans",
) -> tuple[Any, ...]:
    """v32 caption rendering with resolution-proportional margins and font floor."""
    from . import caption_renderer
    from .production_caption_layout_v32 import _rendered_caption_bounds

    cues = caption_renderer.build_caption_cues(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    margin_v = int(round(height * 0.21))
    center_x = width // 2
    baseline_y = height - margin_v
    base_size = max(28, int(round(height * 0.035)))
    horizontal_margin = proportional_caption_margin_v67(width)
    outline = max(2, int(round(3 * width / 1080)))

    layouts = [
        fit_caption_lines_v67(
            cue.text,
            width=width,
            height=height,
            font_name=font_name,
            horizontal_margin=horizontal_margin,
            outline=outline,
        )
        for cue in cues
    ]
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
        rendered = r"\N".join(caption_renderer._escape_ass(line) for line in layout["lines"])
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
    maximum_width = max((int(layout["maximum_line_width_pixels"]) for layout in layouts), default=0)
    safe_width = min((int(layout["safe_width_pixels"]) for layout in layouts), default=0)
    manifest = {
        "format": "ass",
        "rendering": "separate_animated_caption_layer",
        "layout_version": "v67-resolution-proportional-two-line",
        "width": width,
        "height": height,
        "font_name": font_name,
        "font_size": base_size,
        "safe_zone": "bottom_21_percent_baseline; proportional horizontal margin",
        "horizontal_safe_margin_pixels": horizontal_margin,
        "safe_line_width_pixels": safe_width,
        "maximum_rendered_line_width_pixels": maximum_width,
        "all_cues_fit": bool(layouts) and maximum_width <= safe_width,
        "all_rendered_cues_fit": bool(rendered_bounds) and len(rendered_bounds) == len(cues),
        "cues": manifest_cues,
    }
    if not manifest["all_cues_fit"] or not manifest["all_rendered_cues_fit"]:
        raise caption_renderer.CaptionRenderError("Caption pixel-fit verification failed")
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return cues


def install_production_caption_scale_v67() -> None:
    """Install resolution-proportional caption layout for exact preflight and final render."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import caption_renderer
    caption_renderer.write_animated_caption_track = write_scaled_caption_track_v67
    _INSTALLED = True
