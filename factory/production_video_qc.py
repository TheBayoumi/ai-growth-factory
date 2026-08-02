from __future__ import annotations

import contextvars
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from . import video_qc


_MEDIA_TYPES: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "production_video_qc_media_types", default=()
)
_CAPTION_SAFE_RATIO = 0.68
_ORIGINAL_RAW_GRAY_FRAMES = video_qc._raw_gray_frames
_ORIGINAL_VERIFY = video_qc.verify_video_output
_INSTALLED = False


def _upper_visual_frames(
    video: Any,
    *,
    center: float,
    sample_seconds: float = 1.0,
) -> list[bytes]:
    """Exclude the lower caption layer from temporal motion analysis."""
    frames = _ORIGINAL_RAW_GRAY_FRAMES(
        video,
        center=center,
        sample_seconds=sample_seconds,
    )
    source_width = 90
    source_height = 160
    kept_height = max(1, round(source_height * _CAPTION_SAFE_RATIO))
    kept_size = source_width * kept_height
    return [frame[:kept_size] for frame in frames]


def _production_temporal_stability(
    video: Any,
    duration: float,
    scene_durations: Sequence[float] | None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
    means: list[float] = []
    near_static_ratios: list[float] = []
    jump_ratios: list[float] = []
    maxima: list[float] = []
    stutter_windows = 0
    media_types = _MEDIA_TYPES.get()
    centers = video_qc._scene_centers(duration, scene_durations)
    for index, center in enumerate(centers):
        frames = video_qc._raw_gray_frames(video, center=center)
        differences = [
            video_qc._mean_abs_difference(first, second)
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
        media_type = media_types[index] if index < len(media_types) else "video"
        if media_type == "video" and video_qc._is_hold_jump_stutter(differences):
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


def _infer_scene_media_types(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, ...]:
    explicit = kwargs.pop("scene_media_types", None)
    if explicit is not None:
        return tuple(str(value).strip().lower() for value in explicit)
    video_path = kwargs.get("video_path")
    if video_path is None and len(args) >= 2:
        video_path = args[1]
    try:
        path = Path(video_path)
        manifest = path.parent.parent / "scene-media" / "scene-media-manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assets = sorted(
            data.get("assets") or [],
            key=lambda item: int(item.get("scene_index", -1)),
        )
        return tuple(
            str(item.get("media_type") or "").strip().lower()
            for item in assets
        )
    except Exception:
        return ()


def verify_production_video_output(
    *args: Any,
    scene_media_types: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Apply strict motion QC only to generated video scenes, not intentional stills."""
    normalized = tuple(str(value).strip().lower() for value in (scene_media_types or ()))
    if any(value not in {"image", "video"} for value in normalized):
        raise video_qc.VideoQCError(f"Unsupported scene media type sequence: {normalized}")
    token = _MEDIA_TYPES.set(normalized)
    try:
        return _ORIGINAL_VERIFY(*args, **kwargs)
    finally:
        _MEDIA_TYPES.reset(token)


def install_production_video_qc() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    video_qc._raw_gray_frames = _upper_visual_frames
    video_qc._temporal_stability = _production_temporal_stability

    def installed_verify(*args: Any, **kwargs: Any) -> Any:
        media_types = _infer_scene_media_types(args, kwargs)
        return verify_production_video_output(
            *args,
            scene_media_types=media_types,
            **kwargs,
        )

    video_qc.verify_video_output = installed_verify
    _INSTALLED = True
