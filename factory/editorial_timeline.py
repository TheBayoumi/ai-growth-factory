from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from .models import NarrationSegment, VideoPackage
from .video_profile import VideoProfile
from .visual_prompt import SceneVisualPrompt, VisualPlan


@dataclass(frozen=True)
class ShotSpec:
    shot_id: int
    beat_index: int
    source_index: int
    start_seconds: float
    duration_seconds: float
    renderer: str
    semantic_claim: str
    visual_direction: str
    treatment: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_TREATMENTS = (
    "tight concrete detail with a clear foreground action",
    "wide contextual view showing the surrounding workflow",
    "cause-to-result process view with visible directional change",
    "human-scale consequence using generic unbranded people or tools when relevant",
    "clean comparison view with two visibly different arrangements",
)


def _segment_durations(
    segments: Sequence[NarrationSegment], total_duration: float
) -> list[float]:
    ordered = sorted(segments, key=lambda item: item.segment_id)
    durations: list[float] = []
    for index, segment in enumerate(ordered):
        end = ordered[index + 1].start_seconds if index + 1 < len(ordered) else total_duration
        durations.append(max(0.5, float(end) - float(segment.start_seconds)))
    scale = total_duration / max(sum(durations), 0.001)
    return [duration * scale for duration in durations]


def _shot_counts(durations: list[float], profile: VideoProfile) -> list[int]:
    counts = [max(2, math.ceil(value / profile.maximum_shot_seconds)) for value in durations]
    counts[0] = max(
        counts[0],
        profile.first_ten_seconds_minimum_shots,
        math.ceil(durations[0] / profile.maximum_wan_shot_seconds),
    )

    while sum(counts) < profile.minimum_shots:
        index = max(range(len(counts)), key=lambda item: durations[item] / counts[item])
        counts[index] += 1
    while sum(counts) < profile.target_shots:
        candidates = [
            index
            for index in range(len(counts))
            if durations[index] / (counts[index] + 1) >= profile.minimum_shot_seconds
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: durations[item] / counts[item])
        counts[index] += 1
    if sum(counts) > profile.maximum_shots:
        raise ValueError(
            f"Editorial timeline needs {sum(counts)} shots, above configured maximum "
            f"{profile.maximum_shots}"
        )
    if any(
        duration / count > profile.maximum_shot_seconds + 1e-6
        for duration, count in zip(durations, counts, strict=True)
    ):
        raise ValueError("Editorial timeline contains an overlong shot")
    return counts


def _wan_indices(shots: list[ShotSpec], profile: VideoProfile) -> set[int]:
    selected = {0}
    eligible = [
        shot
        for shot in shots[1:]
        if shot.duration_seconds <= profile.maximum_wan_shot_seconds
    ]
    if len(eligible) < profile.wan_shots - 1:
        raise ValueError("Not enough short shots are available for the configured Wan budget")
    duration = sum(shot.duration_seconds for shot in shots)
    targets = [duration * fraction for fraction in (0.52, 0.86)]
    for target in targets[: profile.wan_shots - 1]:
        candidate = min(
            (shot for shot in eligible if shot.shot_id not in selected),
            key=lambda shot: abs(shot.start_seconds - target),
        )
        selected.add(candidate.shot_id)
    if len(selected) != profile.wan_shots:
        raise ValueError("Could not select the exact Wan shot budget")
    return selected


def build_editorial_plan(
    *,
    plan: VisualPlan,
    package: VideoPackage,
    segments: Sequence[NarrationSegment],
    total_duration: float,
    profile: VideoProfile,
) -> tuple[VisualPlan, tuple[ShotSpec, ...]]:
    if len(plan.scenes) != len(package.scenes) or len(segments) != len(package.scenes):
        raise ValueError("Visual plan, package scenes, and narration beats must align")
    durations = _segment_durations(segments, total_duration)
    counts = _shot_counts(durations, profile)

    preliminary: list[ShotSpec] = []
    cursor = 0.0
    shot_id = 0
    for beat_index, (beat_duration, count) in enumerate(zip(durations, counts, strict=True)):
        base = beat_duration / count
        for local_index in range(count):
            duration = beat_duration - base * (count - 1) if local_index + 1 == count else base
            package_scene = package.scenes[beat_index]
            treatment = _TREATMENTS[(beat_index + local_index) % len(_TREATMENTS)]
            seed_material = f"{package.title}|{beat_index}|{local_index}|v28".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big") & 0x7FFFFFFF
            preliminary.append(
                ShotSpec(
                    shot_id=shot_id,
                    beat_index=beat_index,
                    source_index=package_scene.source_index,
                    start_seconds=round(cursor, 6),
                    duration_seconds=round(duration, 6),
                    renderer="parallax",
                    semantic_claim=f"{package_scene.heading}. {package_scene.body}",
                    visual_direction=package_scene.visual,
                    treatment=treatment,
                    seed=seed,
                )
            )
            cursor += duration
            shot_id += 1

    wan = _wan_indices(preliminary, profile)
    shots = tuple(
        replace(shot, renderer="wan_i2v" if shot.shot_id in wan else "parallax")
        for shot in preliminary
    )
    if sum(shot.start_seconds < 10.0 for shot in shots) < profile.first_ten_seconds_minimum_shots:
        raise ValueError("Opening does not contain enough unique shots")

    expanded_scenes: list[SceneVisualPrompt] = []
    for shot in shots:
        source = plan.scenes[shot.beat_index]
        prompt = (
            f"Factual technology documentary shot for this exact claim: {shot.semantic_claim}. "
            f"Required visual subject and context: {shot.visual_direction}. "
            f"Shot treatment: {shot.treatment}. Depict the concrete idea literally. Generic "
            "unbranded researchers, workspaces, tools, devices, code-like structures, or "
            "procedural diagrams are allowed when they communicate the claim. Preserve a "
            "full-frame environment and keep only the immediate caption area moderately calm; "
            "never replace the factual subject with generic corridors, towers, blocks, or orbs."
        )
        motion = (
            f"Animate the concrete subject for beat {shot.beat_index} with one meaningful "
            f"change that supports this claim: {shot.semantic_claim}. Keep identities and "
            "geometry stable, use restrained documentary camera movement, and introduce no cut."
        )
        expanded_scenes.append(
            replace(
                source,
                scene_index=shot.shot_id,
                source_index=shot.source_index,
                generation_mode="wan_i2v" if shot.renderer == "wan_i2v" else "image",
                image_prompt=prompt,
                motion_prompt=motion,
                continuity_anchor=(
                    f"secondary palette and lighting accent for beat {shot.beat_index}; "
                    "never reuse the primary subject"
                ),
                seed=shot.seed,
                duration_seconds=shot.duration_seconds,
            )
        )
    expanded = replace(
        plan,
        prompt_version="visual-director-v28-editorial",
        scenes=tuple(expanded_scenes),
    )
    return expanded, shots
