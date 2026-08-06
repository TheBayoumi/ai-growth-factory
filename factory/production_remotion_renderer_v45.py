from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from .editorial_timeline import ShotSpec
from .models import NarrationSegment, VideoPackage
from .remotion_bridge import render_with_remotion
from .remotion_contract import build_remotion_render_spec
from .video_profile import VideoProfile


_INSTALLED = False


def _enabled() -> bool:
    return os.getenv("VIDEO_RENDER_BACKEND", "ffmpeg").strip().lower() == "remotion"


def compose_editorial_video_remotion_v45(
    *,
    media: Sequence[Any],
    shots: Sequence[ShotSpec],
    segments: Sequence[NarrationSegment],
    package: VideoPackage,
    audio_path: Path,
    workdir: Path,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, Path, Path]:
    """Render the reviewed timeline through Remotion without changing media inference."""
    from . import caption_renderer, visual_compositor
    from .production_editorial_compositor_v28 import _write_ambient_music_bed
    from .production_editorial_v28 import _audio_duration

    profile = VideoProfile.from_env()
    ordered_media = sorted(media, key=lambda item: item.scene_index)
    ordered_shots = sorted(shots, key=lambda item: item.shot_id)
    if not profile.minimum_shots <= len(ordered_shots) <= profile.maximum_shots:
        raise ValueError(
            f"Editorial shot count {len(ordered_shots)} is outside the rendered profile "
            f"{profile.minimum_shots}-{profile.maximum_shots}"
        )
    if len(ordered_media) != len(ordered_shots):
        raise ValueError("Every editorial shot requires one unique media asset")
    if [item.scene_index for item in ordered_media] != list(range(len(ordered_media))):
        raise ValueError("Editorial media indices are not contiguous")
    if [item.shot_id for item in ordered_shots] != list(range(len(ordered_shots))):
        raise ValueError("Editorial shot IDs are not contiguous")
    if len({str(item.path) for item in ordered_media}) != len(ordered_media):
        raise ValueError("Editorial composition cannot reuse a media path")
    if any(
        item.duration_seconds < profile.minimum_shot_seconds - 1e-6
        or item.duration_seconds > profile.maximum_shot_seconds + 1e-6
        for item in ordered_shots
    ):
        raise ValueError("Editorial composition contains a shot outside duration bounds")
    wan_assets = sum(str(item.media_type) == "video" for item in ordered_media)
    if wan_assets != profile.wan_shots:
        raise ValueError(
            f"Rendered media contains {wan_assets} Wan shots; profile requires "
            f"{profile.wan_shots}"
        )

    workdir.mkdir(parents=True, exist_ok=True)
    caption_path = workdir / "animated-captions.ass"
    cues = caption_renderer.write_animated_caption_track(
        sorted(segments, key=lambda item: item.segment_id),
        caption_path,
        width=width,
        height=height,
    )
    total_duration = _audio_duration(audio_path)
    planned_duration = sum(item.duration_seconds for item in ordered_shots)
    if abs(total_duration - planned_duration) > 0.08:
        raise ValueError(
            f"Editorial timeline {planned_duration:.3f}s does not match narration "
            f"{total_duration:.3f}s"
        )

    # Preserve the existing deterministic, license-free music contract. Remotion owns the
    # timeline and captions; it must not silently remove reviewed audio layers.
    import imageio_ffmpeg

    music_path = _write_ambient_music_bed(
        workdir / "background-music.wav",
        duration_seconds=total_duration,
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),
    )
    spec = build_remotion_render_spec(
        shots=ordered_shots,
        media=ordered_media,
        segments=segments,
        package=package,
        audio_path=audio_path,
        background_music_path=music_path,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=total_duration,
        caption_cues=cues,
    )
    canonical_spec_path = workdir / "remotion-render-spec.json"
    spec.write_json(canonical_spec_path)

    output = workdir / "video.mp4"
    output, remotion_manifest_path, remotion_log_path = render_with_remotion(
        spec=spec,
        output_path=output,
        workdir=workdir,
    )
    compositor_log = workdir / "visual-compositor.log"
    shutil.copy2(remotion_log_path, compositor_log)

    thumbnail = workdir / "thumbnail.png"
    visual_compositor._thumbnail(ordered_media[0].keyframe_path, package, thumbnail)
    remotion_manifest = json.loads(remotion_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "renderer": "remotion_editorial_timeline_v45",
        "renderer_version": remotion_manifest["renderer_version"],
        "editorial_contract": profile.as_dict(),
        "render_spec": str(canonical_spec_path),
        "render_spec_sha256": spec.sha256(),
        "realized_shot_count": len(ordered_shots),
        "realized_wan_shots": wan_assets,
        "source_asset_looping": False,
        "destructive_caption_matte": False,
        "still_motion": "storyboard_camera_motion_remotion_interpolation",
        "transition_frames": spec.transition_frames,
        "background_music": {
            "path": str(music_path),
            "source": "deterministic_license_free_ambient_triads",
            "mixed_beneath_reviewed_narration": True,
        },
        "pixel_format": "yuv420p",
        "constant_frame_rate": fps,
        "caption_layer": str(caption_path),
        "caption_renderer": "remotion_phrase_card",
        "caption_cues": len(cues),
        "shot_count": len(ordered_shots),
        "shots": [item.as_dict() for item in ordered_shots],
        "scene_media": [item.as_dict() for item in ordered_media],
        "remotion_manifest": str(remotion_manifest_path),
        "output": {
            "path": str(output),
            "width": width,
            "height": height,
            "fps": fps,
            "audio_path": str(audio_path),
            "background_music_path": str(music_path),
        },
    }
    (workdir / "visual-composition-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output, thumbnail, caption_path


def install_production_remotion_renderer_v45() -> None:
    """Select Remotion only through an explicit feature flag; FFmpeg stays as A/B fallback."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if not _enabled():
        return

    from . import production_editorial_v28
    from . import production_editorial_compositor_v28

    production_editorial_v28._compose_editorial_video = (
        compose_editorial_video_remotion_v45
    )
    production_editorial_compositor_v28.compose_editorial_video_v28 = (
        compose_editorial_video_remotion_v45
    )
