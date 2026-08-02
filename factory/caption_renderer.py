from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .models import NarrationSegment


class CaptionRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptionCue:
    cue_id: int
    scene_index: int
    start_seconds: float
    end_seconds: float
    text: str
    word_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _escape_ass(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )


def _phrase_chunks(text: str, *, maximum_words: int = 6) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        punctuation_break = bool(re.search(r"[,;:!?]$", word)) and len(current) >= 3
        if len(current) >= maximum_words or punctuation_break:
            chunks.append(" ".join(current))
            current = []
    if current:
        if chunks and len(current) <= 2 and len(chunks[-1].split()) + len(current) <= maximum_words + 1:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


def _segment_bounds(segment: NarrationSegment, next_start: float | None) -> tuple[float, float]:
    start = max(0.0, float(segment.start_seconds))
    end = float(segment.end_seconds)
    if end <= start and next_start is not None:
        end = next_start
    if end <= start:
        raise CaptionRenderError(f"Narration segment {segment.segment_id} has invalid timing")
    return start, end


def build_caption_cues(segments: Sequence[NarrationSegment]) -> tuple[CaptionCue, ...]:
    ordered = sorted(segments, key=lambda segment: segment.segment_id)
    cues: list[CaptionCue] = []
    cue_id = 0
    for index, segment in enumerate(ordered):
        next_start = (
            float(ordered[index + 1].start_seconds)
            if index + 1 < len(ordered)
            else None
        )
        start, end = _segment_bounds(segment, next_start)
        chunks = _phrase_chunks(segment.text)
        if not chunks:
            raise CaptionRenderError(f"Narration segment {segment.segment_id} is empty")
        weights = [max(1, len(chunk.split())) for chunk in chunks]
        total_weight = sum(weights)
        cursor = start
        for chunk_index, (chunk, weight) in enumerate(zip(chunks, weights, strict=True)):
            duration = (end - start) * weight / total_weight
            cue_end = end if chunk_index + 1 == len(chunks) else cursor + duration
            if cue_end - cursor < 0.32:
                raise CaptionRenderError(
                    f"Caption cue is too short in segment {segment.segment_id}: {cue_end - cursor:.3f}s"
                )
            cues.append(
                CaptionCue(
                    cue_id=cue_id,
                    scene_index=segment.segment_id,
                    start_seconds=round(cursor, 3),
                    end_seconds=round(cue_end, 3),
                    text=chunk,
                    word_count=len(chunk.split()),
                )
            )
            cursor = cue_end
            cue_id += 1
    return tuple(cues)


def _karaoke_text(cue: CaptionCue) -> str:
    words = cue.text.split()
    duration_cs = max(1, int(round((cue.end_seconds - cue.start_seconds) * 100)))
    base, remainder = divmod(duration_cs, len(words))
    pieces: list[str] = []
    for index, word in enumerate(words):
        word_duration = base + (1 if index < remainder else 0)
        pieces.append(rf"{{\kf{max(1, word_duration)}}}{_escape_ass(word)}")
    return " ".join(pieces)


def write_animated_caption_track(
    segments: Sequence[NarrationSegment],
    output_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "DejaVu Sans",
) -> tuple[CaptionCue, ...]:
    """Write phrase-level animated captions as a separate ASS subtitle layer."""
    cues = build_caption_cues(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_size = max(42, int(round(height * 0.035)))
    margin_v = int(round(height * 0.21))
    center_x = width // 2
    baseline_y = height - margin_v
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{font_size},&H00FFFFFF,&H0000D7FF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,3,3,0,2,90,90,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for cue in cues:
        animation = (
            rf"{{\an2\pos({center_x},{baseline_y})\fad(70,90)"
            rf"\fscx92\fscy92\t(0,140,\fscx100\fscy100)}}"
        )
        lines.append(
            "Dialogue: 0,"
            + _ass_time(cue.start_seconds)
            + ","
            + _ass_time(cue.end_seconds)
            + ",Caption,,0,0,0,,"
            + animation
            + _karaoke_text(cue)
            + "\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8")
    manifest = {
        "format": "ass",
        "rendering": "separate_animated_caption_layer",
        "width": width,
        "height": height,
        "font_name": font_name,
        "font_size": font_size,
        "safe_zone": "bottom_21_percent_baseline; platform controls remain below captions",
        "cues": [cue.as_dict() for cue in cues],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cues
