from __future__ import annotations

import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg

from .config import Settings
from .models import NarrationSegment, VideoPackage
from .policy import Strategy


_TRANSITION_SECONDS = 0.18


def _balanced_caption_chunks(text: str) -> list[str]:
    words = text.split()
    if len(words) <= 9:
        return [" ".join(words)]
    chunk_count = min(3, max(2, math.ceil(len(words) / 9)))
    base, remainder = divmod(len(words), chunk_count)
    chunks: list[str] = []
    cursor = 0
    for index in range(chunk_count):
        size = base + (1 if index < remainder else 0)
        chunks.append(" ".join(words[cursor : cursor + size]))
        cursor += size
    return chunks


def _chunk_durations(chunks: list[str], total_seconds: float) -> list[float]:
    weights = [max(1, len(chunk.split())) for chunk in chunks]
    total_weight = sum(weights)
    return [total_seconds * weight / total_weight for weight in weights]


def _motif_hint(scene_text: str, scene_index: int) -> str:
    text = scene_text.casefold()
    if any(word in text for word in ("govern", "ethic", "safe", "secure", "policy")):
        return "verification checklist"
    if any(word in text for word in ("agent", "tool", "workflow", "task", "computer")):
        return "repair loop workflow"
    if any(word in text for word in ("research", "science", "discover", "study")):
        return "source graph"
    if any(word in text for word in ("workforce", "professional", "adoption", "rollout")):
        return "scene timeline"
    return (
        "source graph",
        "scene timeline",
        "verification checklist",
        "repair loop workflow",
        "audio waveform",
        "dashboard",
    )[scene_index % 6]


def _scene_durations(
    segments: Sequence[NarrationSegment],
    audio_duration: float,
) -> list[float]:
    ordered = sorted(segments, key=lambda item: item.segment_id)
    durations: list[float] = []
    for index, segment in enumerate(ordered):
        end = ordered[index + 1].start_seconds if index + 1 < len(ordered) else audio_duration
        durations.append(max(0.5, end - segment.start_seconds))
    scale = audio_duration / max(sum(durations), 0.001)
    return [duration * scale for duration in durations]


def render_platform_video(
    settings: Settings,
    package: VideoPackage,
    strategy: Strategy,
    workdir: Path,
    *,
    segments: Sequence[NarrationSegment] | None = None,
) -> tuple[Path, Path]:
    """Render phrase-synchronized cards with smooth fades and no camera shake."""
    from . import render as base

    if not segments or len(segments) != len(package.scenes):
        return base._original_render_video(  # type: ignore[attr-defined]
            settings,
            package,
            strategy,
            workdir,
            segments=segments,
        )

    workdir.mkdir(parents=True, exist_ok=True)
    audio = workdir / "voice.wav"
    if not audio.exists():
        raise FileNotFoundError("voice.wav must exist before rendering")
    audio_duration = base._wav_duration(audio)
    if audio_duration < 1.0:
        raise ValueError("Narration is too short to render")

    varied_scenes = []
    for index, scene in enumerate(package.scenes):
        scene_text = f"{scene.heading} {scene.body} {scene.visual}"
        varied_scenes.append(replace(scene, visual=_motif_hint(scene_text, index)))
    visual_package = replace(package, scenes=varied_scenes)

    ordered_segments = sorted(segments, key=lambda item: item.segment_id)
    durations = _scene_durations(ordered_segments, audio_duration)
    cards: list[tuple[Path, float]] = []
    for scene_index, (segment, scene_duration) in enumerate(
        zip(ordered_segments, durations, strict=True)
    ):
        chunks = _balanced_caption_chunks(segment.text)
        chunk_durations = _chunk_durations(chunks, scene_duration)
        for chunk_index, (caption, duration) in enumerate(
            zip(chunks, chunk_durations, strict=True)
        ):
            path = workdir / f"scene-{scene_index:02d}-caption-{chunk_index:02d}.png"
            base._scene_card(
                settings,
                visual_package,
                strategy,
                scene_index,
                caption,
                path,
            )
            cards.append((path, duration))

    thumbnail = workdir / "thumbnail.png"
    base._thumbnail(settings, visual_package, strategy, thumbnail)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for index, (image, duration) in enumerate(cards):
        input_duration = duration + (_TRANSITION_SECONDS if index + 1 < len(cards) else 0.0)
        command += ["-loop", "1", "-t", f"{input_duration:.3f}", "-i", str(image)]
    command += ["-i", str(audio)]

    filters: list[str] = []
    for index, (_image, duration) in enumerate(cards):
        input_duration = duration + (_TRANSITION_SECONDS if index + 1 < len(cards) else 0.0)
        filters.append(
            f"[{index}:v]scale={settings.width}:{settings.height},"
            f"fps={settings.fps},trim=duration={input_duration:.3f},"
            "settb=AVTB,setpts=PTS-STARTPTS,setsar=1,format=yuv420p"
            f"[v{index}]"
        )

    if len(cards) == 1:
        output_label = "v0"
    else:
        cumulative = cards[0][1]
        previous = "v0"
        for index in range(1, len(cards)):
            output_label = f"x{index}"
            filters.append(
                f"[{previous}][v{index}]xfade=transition=fade:"
                f"duration={_TRANSITION_SECONDS:.3f}:offset={cumulative:.3f}"
                f"[{output_label}]"
            )
            previous = output_label
            cumulative += cards[index][1]
        output_label = previous

    output = workdir / "video.mp4"
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{output_label}]",
        "-map",
        f"{len(cards)}:a",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-profile:v",
        "high",
        "-level",
        "4.1",
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
        timeout=420,
        check=False,
    )
    if completed.returncode != 0 or not output.exists() or output.stat().st_size < 100_000:
        raise RuntimeError(f"Platform render failed: {completed.stderr[-3000:]}")
    return output, thumbnail


def install_production_renderer() -> None:
    from . import render

    if hasattr(render, "_original_render_video"):
        return
    render._original_render_video = render.render_video  # type: ignore[attr-defined]
    render.render_video = render_platform_video
