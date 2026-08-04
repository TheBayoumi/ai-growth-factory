from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from .models import NarrationSegment, VideoPackage
from .video_profile import VideoProfile
from .visual_prompt import SceneVisualPrompt, VisualPlan


@dataclass(frozen=True)
class StoryBeat:
    beat_id: int
    segment_id: int
    sentence_index: int
    start_seconds: float
    duration_seconds: float
    narration_text: str
    scene_candidates: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scene_candidates"] = list(self.scene_candidates)
        return payload


@dataclass(frozen=True)
class ShotSpec:
    shot_id: int
    beat_index: int
    segment_id: int
    package_scene_index: int
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
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_CONCEPT_PREFIXES = (
    ("collaborat", "collaboration"),
    ("community", "collaboration"),
    ("shared", "collaboration"),
    ("share", "collaboration"),
    ("scalab", "scale"),
    ("scalability", "scale"),
    ("infrastruct", "infrastructure"),
    ("perform", "performance"),
    ("agentic", "agent"),
    ("agents", "agent"),
    ("evaluat", "evaluation"),
    ("training", "training"),
)
_STOPWORDS = {
    "about", "across", "after", "also", "among", "and", "are", "because", "been",
    "before", "being", "between", "both", "can", "could", "does", "each", "for",
    "from", "has", "have", "into", "its", "more", "most", "new", "not", "of", "on",
    "one", "only", "or", "other", "our", "over", "that", "the", "their", "these",
    "this", "through", "to", "using", "was", "were", "which", "while", "with", "would",
}


def _normalize_token(token: str) -> str:
    for prefix, replacement in _CONCEPT_PREFIXES:
        if token.startswith(prefix):
            return replacement
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ers") and len(token) > 5:
        return token[:-1]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    if token.endswith("ed") and len(token) > 5:
        return token[:-1]
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    return token


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        _normalize_token(token)
        for token in _TOKEN_RE.findall(value.casefold())
        if token not in _STOPWORDS
    )


def _scene_candidates(text: str, package: VideoPackage) -> tuple[int, ...]:
    beat_tokens = _tokens(text)
    ranked: list[tuple[float, int]] = []
    for index, scene in enumerate(package.scenes):
        scene_tokens = _tokens(f"{scene.heading} {scene.body} {scene.visual}")
        intersection = beat_tokens & scene_tokens
        union = beat_tokens | scene_tokens
        score = len(intersection) * 3.0 + (len(intersection) / len(union) if union else 0.0)
        ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best = ranked[0][0] if ranked else 0.0
    selected = [index for score, index in ranked if score > 0.0 and score >= best * 0.55][:3]
    return tuple(selected or ([ranked[0][1]] if ranked else [0]))


def _story_beats(
    segments: Sequence[NarrationSegment],
    total_duration: float,
    package: VideoPackage,
) -> list[StoryBeat]:
    ordered = sorted(segments, key=lambda item: item.segment_id)
    beats: list[StoryBeat] = []
    beat_id = 0
    for segment_index, segment in enumerate(ordered):
        start = max(0.0, float(segment.start_seconds))
        end = (
            float(ordered[segment_index + 1].start_seconds)
            if segment_index + 1 < len(ordered)
            else total_duration
        )
        end = max(start + 0.25, end)
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(segment.text.strip())
            if sentence.strip()
        ] or [segment.text.strip()]
        weights = [max(1, len(sentence.split())) for sentence in sentences]
        total_weight = sum(weights)
        cursor = start
        for sentence_index, (sentence, weight) in enumerate(zip(sentences, weights, strict=True)):
            duration = (end - start) * weight / total_weight
            beat_end = end if sentence_index + 1 == len(sentences) else cursor + duration
            beats.append(
                StoryBeat(
                    beat_id=beat_id,
                    segment_id=segment.segment_id,
                    sentence_index=sentence_index,
                    start_seconds=round(cursor, 6),
                    duration_seconds=round(beat_end - cursor, 6),
                    narration_text=sentence,
                    scene_candidates=_scene_candidates(sentence, package),
                )
            )
            cursor = beat_end
            beat_id += 1
    if not beats:
        raise ValueError("Narration produced no editorial story beats")
    return beats


def _opening_shot_count(beats: list[StoryBeat], counts: list[int]) -> int:
    total = 0
    for beat, count in zip(beats, counts, strict=True):
        step = beat.duration_seconds / count
        total += sum(beat.start_seconds + step * index < 10.0 for index in range(count))
    return total


def _shot_counts(beats: list[StoryBeat], profile: VideoProfile) -> list[int]:
    counts = [max(1, math.ceil(beat.duration_seconds / profile.maximum_shot_seconds)) for beat in beats]
    counts[0] = max(
        counts[0],
        math.ceil(beats[0].duration_seconds / profile.maximum_wan_shot_seconds),
    )

    while _opening_shot_count(beats, counts) < profile.first_ten_seconds_minimum_shots:
        candidates = [
            index
            for index, beat in enumerate(beats)
            if beat.start_seconds < 10.0
            and beat.duration_seconds / (counts[index] + 1) >= profile.minimum_shot_seconds
        ]
        if not candidates:
            raise ValueError("Opening cannot satisfy the unique-shot requirement")
        index = max(candidates, key=lambda item: beats[item].duration_seconds / counts[item])
        counts[index] += 1

    while sum(counts) < profile.minimum_shots:
        candidates = [
            index
            for index, beat in enumerate(beats)
            if beat.duration_seconds / (counts[index] + 1) >= profile.minimum_shot_seconds
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: beats[item].duration_seconds / counts[item])
        counts[index] += 1
    while sum(counts) < profile.target_shots:
        candidates = [
            index
            for index, beat in enumerate(beats)
            if beat.duration_seconds / (counts[index] + 1) >= profile.minimum_shot_seconds
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: beats[item].duration_seconds / counts[item])
        counts[index] += 1
    if sum(counts) > profile.maximum_shots:
        raise ValueError(
            f"Editorial timeline needs {sum(counts)} shots, above configured maximum "
            f"{profile.maximum_shots}"
        )
    if any(
        beat.duration_seconds / count > profile.maximum_shot_seconds + 1e-6
        for beat, count in zip(beats, counts, strict=True)
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
    if len(plan.scenes) != len(package.scenes):
        raise ValueError("Visual plan and package scenes must align")
    if not segments:
        raise ValueError("Narration segments are missing")
    beats = _story_beats(segments, total_duration, package)
    counts = _shot_counts(beats, profile)

    preliminary: list[ShotSpec] = []
    shot_id = 0
    for beat, count in zip(beats, counts, strict=True):
        base = beat.duration_seconds / count
        for local_index in range(count):
            duration = (
                beat.duration_seconds - base * (count - 1)
                if local_index + 1 == count
                else base
            )
            scene_index = beat.scene_candidates[local_index % len(beat.scene_candidates)]
            package_scene = package.scenes[scene_index]
            treatment = _TREATMENTS[(beat.beat_id + local_index) % len(_TREATMENTS)]
            seed_material = f"{package.title}|{beat.beat_id}|{local_index}|v28".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big") & 0x7FFFFFFF
            preliminary.append(
                ShotSpec(
                    shot_id=shot_id,
                    beat_index=beat.beat_id,
                    segment_id=beat.segment_id,
                    package_scene_index=scene_index,
                    source_index=package_scene.source_index,
                    start_seconds=round(beat.start_seconds + base * local_index, 6),
                    duration_seconds=round(duration, 6),
                    renderer="parallax",
                    semantic_claim=beat.narration_text,
                    visual_direction=package_scene.visual,
                    treatment=treatment,
                    seed=seed,
                )
            )
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
        source = plan.scenes[shot.package_scene_index]
        package_scene = package.scenes[shot.package_scene_index]
        prompt = (
            f"Factual technology documentary shot synchronized to this exact spoken sentence: "
            f"{shot.semantic_claim}. Supporting source-grounded visual direction: "
            f"{shot.visual_direction}. Shot treatment: {shot.treatment}. Depict the concrete "
            "idea literally. Generic unbranded researchers, workspaces, tools, devices, "
            "code-like structures, or procedural diagrams are allowed when they communicate "
            "the spoken claim. Preserve a full-frame environment and keep only the immediate "
            "caption area moderately calm; never replace the factual subject with generic "
            "corridors, towers, blocks, or orbs."
        )
        motion = (
            f"Animate the concrete subject for spoken beat {shot.beat_index} with one meaningful "
            f"change that supports this sentence: {shot.semantic_claim}. Keep identities and "
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
                    f"secondary palette and lighting accent for {package_scene.heading}; "
                    "never reuse the primary subject"
                ),
                seed=shot.seed,
                duration_seconds=shot.duration_seconds,
            )
        )
    expanded = replace(
        plan,
        prompt_version="visual-director-v28-editorial-beat-aligned",
        scenes=tuple(expanded_scenes),
    )
    return expanded, shots
