from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import imageio_ffmpeg

from .editorial_timeline import ShotSpec
from .models import NarrationSegment, VideoPackage
from .video_profile import VideoProfile


_TRANSITION_SECONDS = 0.16


def _filter_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


def compose_editorial_video_v28(
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
    """Compose unique editorial shots without looping any source video.

    Every input is normalized to a constant 1/fps timebase before xfade. Image shots receive
    deterministic Ken Burns motion. Wan shots may be padded only by the crossfade duration,
    never replayed. The exact profile used by the renderer is persisted in the composition
    manifest so CI verifies the generated contract instead of duplicating mutable constants.
    """
    from . import caption_renderer, visual_compositor
    from .production_editorial_v28 import _audio_duration

    profile = VideoProfile.from_env()
    ordered_media = sorted(media, key=lambda item: item.scene_index)
    ordered_shots = sorted(shots, key=lambda item: item.shot_id)
    if not ordered_shots:
        raise ValueError("The editorial timeline contains no shots")
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
    if any(item.duration_seconds <= 0 for item in ordered_shots):
        raise ValueError("Editorial shot duration must be positive")
    if any(item.duration_seconds > profile.maximum_shot_seconds + 1e-6 for item in ordered_shots):
        raise ValueError("Editorial composition contains an overlong shot")
    if len({str(item.path) for item in ordered_media}) != len(ordered_media):
        raise ValueError("Editorial composition cannot reuse a media path")
    wan_assets = sum(str(item.media_type) == "video" for item in ordered_media)
    if wan_assets != profile.wan_shots:
        raise ValueError(
            f"Rendered media contains {wan_assets} Wan shots; profile requires {profile.wan_shots}"
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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for asset, shot in zip(ordered_media, ordered_shots, strict=True):
        input_duration = shot.duration_seconds + _TRANSITION_SECONDS
        if asset.media_type == "image":
            command += [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{input_duration:.6f}",
                "-i",
                str(asset.path),
            ]
        elif asset.media_type == "video":
            command += ["-i", str(asset.path)]
        else:
            raise ValueError(f"Unsupported editorial media type: {asset.media_type}")
    audio_index = len(ordered_media)
    command += ["-i", str(audio_path)]

    filters: list[str] = []
    for index, (asset, shot) in enumerate(zip(ordered_media, ordered_shots, strict=True)):
        input_duration = shot.duration_seconds + _TRANSITION_SECONDS
        common = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},setsar=1,format=yuv420p"
        )
        timing = (
            f"trim=duration={input_duration:.6f},"
            f"settb=expr=1/{fps},setpts=N/({fps}*TB),fps={fps},"
            "setsar=1,format=yuv420p"
        )
        if asset.media_type == "image":
            zoom = (
                "min(zoom+0.00055,1.055)"
                if index % 2 == 0
                else "max(1.055-0.00055*on,1.0)"
            )
            frame_count = max(1, round(input_duration * fps))
            x = "iw/2-(iw/zoom/2)" if index % 3 else f"(iw-iw/zoom)*on/{frame_count}"
            filters.append(
                f"[{index}:v]{common},"
                f"zoompan=z='{zoom}':x='{x}':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={width}x{height}:fps={fps},{timing}[v{index}]"
            )
        else:
            filters.append(
                f"[{index}:v]{common},"
                f"tpad=stop_mode=clone:stop_duration={_TRANSITION_SECONDS:.3f},"
                f"{timing}[v{index}]"
            )

    previous = "v0"
    cumulative = ordered_shots[0].duration_seconds
    for index in range(1, len(ordered_shots)):
        label = f"x{index}"
        filters.append(
            f"[{previous}][v{index}]xfade=transition=fade:"
            f"duration={_TRANSITION_SECONDS:.3f}:offset={cumulative:.6f}[{label}]"
        )
        previous = label
        cumulative += ordered_shots[index].duration_seconds

    subtitles = _filter_path(caption_path)
    filters.append(
        f"[{previous}]subtitles=filename='{subtitles}':"
        "fontsdir='/usr/share/fonts/truetype/dejavu',format=yuv420p[vout]"
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
        "-pix_fmt",
        "yuv420p",
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
        timeout=900,
        check=False,
    )
    log = workdir / "visual-compositor.log"
    log.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if "-stream_loop" in command:
        raise RuntimeError("v35 compositor attempted source-video looping")
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 500_000:
        raise RuntimeError(f"v35 editorial composition failed: {completed.stderr[-3000:]}")

    thumbnail = workdir / "thumbnail.png"
    visual_compositor._thumbnail(ordered_media[0].keyframe_path, package, thumbnail)
    manifest = {
        "renderer": "ffmpeg_editorial_timeline_v35_profile_driven",
        "editorial_contract": profile.as_dict(),
        "realized_shot_count": len(ordered_shots),
        "realized_wan_shots": wan_assets,
        "source_asset_looping": False,
        "destructive_caption_matte": False,
        "still_motion": "deterministic_ken_burns",
        "pixel_format": "yuv420p",
        "constant_frame_rate": fps,
        "caption_layer": str(caption_path),
        "caption_cues": len(cues),
        "shot_count": len(ordered_shots),
        "shots": [item.as_dict() for item in ordered_shots],
        "scene_media": [item.as_dict() for item in ordered_media],
        "output": {
            "path": str(output),
            "width": width,
            "height": height,
            "fps": fps,
            "audio_path": str(audio_path),
        },
    }
    (workdir / "visual-composition-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output, thumbnail, caption_path
