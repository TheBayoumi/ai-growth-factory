from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Settings
from .feeds import SourceItem
from .models import NarrationSegment, VideoPackage
from .policy import Strategy
from .video_profile import VideoProfile
from .visual_prompt import SceneVisualPrompt, VisualPlan


VIMAX_COMMIT = "05a48943878312d88fe5a016c12a9654940ecc43"
VIMAX_PROMPT_VERSION = f"vimax-script2video@{VIMAX_COMMIT}"
_PLAN_ARTIFACTS: dict[str, Path] = {}


class ViMaxPlanningError(RuntimeError):
    """ViMax planning failed or returned a plan that violates the factory contract."""


@dataclass(frozen=True)
class ViMaxShot:
    idx: int
    cam_idx: int
    visual_desc: str
    first_frame_prompt: str
    last_frame_prompt: str
    motion_prompt: str
    variation_type: str
    variation_reason: str
    audio_desc: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, expected_idx: int) -> "ViMaxShot":
        try:
            idx = int(value["idx"])
            cam_idx = int(value["cam_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ViMaxPlanningError("ViMax shot requires integer idx and cam_idx") from exc
        if idx != expected_idx:
            raise ViMaxPlanningError(
                f"ViMax shot indices must be contiguous; expected {expected_idx}, got {idx}"
            )

        def required(name: str) -> str:
            text = " ".join(str(value.get(name) or "").split()).strip()
            if not text:
                raise ViMaxPlanningError(f"ViMax shot {idx} has empty {name}")
            return text

        variation = required("variation_type").casefold()
        if variation not in {"small", "medium", "large"}:
            raise ViMaxPlanningError(
                f"ViMax shot {idx} has unsupported variation_type {variation!r}"
            )
        return cls(
            idx=idx,
            cam_idx=cam_idx,
            visual_desc=required("visual_desc"),
            first_frame_prompt=required("ff_desc"),
            last_frame_prompt=required("lf_desc"),
            motion_prompt=required("motion_desc"),
            variation_type=variation,
            variation_reason=required("variation_reason"),
            audio_desc=" ".join(str(value.get("audio_desc") or "").split()).strip(),
        )


@dataclass(frozen=True)
class ViMaxPlan:
    shots: tuple[ViMaxShot, ...]
    characters: tuple[dict[str, Any], ...]
    camera_tree: tuple[dict[str, Any], ...]
    artifact_path: Path
    payload_sha256: str


def _planner_script() -> Path:
    configured = os.getenv("VIMAX_PLANNER_SCRIPT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "scripts" / "run_vimax_planner.py").resolve()


def _planner_python() -> str:
    return os.getenv("VIMAX_PYTHON", sys.executable).strip() or sys.executable


def _vimax_root() -> Path:
    configured = os.getenv("VIMAX_ROOT", "").strip()
    if not configured:
        raise ViMaxPlanningError(
            "VIMAX_PLANNER_ENABLED=true requires VIMAX_ROOT pointing to a checkout of "
            f"HKUDS/ViMax at {VIMAX_COMMIT}"
        )
    root = Path(configured).expanduser().resolve()
    if not (root / "pipelines" / "script2video_pipeline.py").is_file():
        raise ViMaxPlanningError(f"VIMAX_ROOT is not a ViMax checkout: {root}")
    return root


def _source_context(package: VideoPackage, sources: Sequence[SourceItem]) -> list[dict[str, Any]]:
    by_url = {item.url: item for item in sources}
    context: list[dict[str, Any]] = []
    for source_index, url in enumerate(package.source_urls):
        item = by_url.get(url)
        if item is None:
            raise ViMaxPlanningError(f"selected source is missing from context: {url}")
        context.append(
            {
                "source_index": source_index,
                "publisher": item.publisher,
                "title": item.title,
                "summary": item.summary,
                "url": item.url,
            }
        )
    return context


def _request_payload(
    *,
    settings: Settings,
    package: VideoPackage,
    sources: Sequence[SourceItem],
    strategy: Strategy,
    output_dir: Path,
) -> dict[str, Any]:
    profile = VideoProfile.from_env()
    api_key = settings.llm_api_key or os.getenv("VIMAX_LLM_API_KEY", "not-needed")
    return {
        "schema_version": "agf-vimax-planner-v1",
        "vimax_commit": VIMAX_COMMIT,
        "script": package.narration,
        "style": (
            "photorealistic vertical technology documentary; factual, cinematic, "
            "unbranded, text-free, coherent recurring subjects and environments"
        ),
        "user_requirement": (
            f"Create {profile.target_shots} shots, with an allowed range of "
            f"{profile.minimum_shots}-{profile.maximum_shots}. Format every shot for a "
            f"vertical 9:16 short lasting {profile.minimum_video_seconds:.0f}-"
            f"{profile.maximum_video_seconds:.0f} seconds. Use at least "
            f"{profile.first_ten_seconds_minimum_shots} distinct shots in the first ten "
            "seconds. Preserve every factual claim and source relationship. Describe an "
            "independent first frame, last frame, physical motion, camera angle, camera "
            "movement, visible subject position, environment, lighting, and continuity. "
            "Never request readable text, logos, watermarks, branded interfaces, article "
            "screenshots, real-person likenesses, morphing, camera shake, or collage layouts."
        ),
        "chat_model": {
            "model_provider": "openai",
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "api_key": api_key,
            "temperature": min(float(settings.llm_temperature), 0.45),
        },
        "factory": {
            "topic": package.topic,
            "title": package.title,
            "scenes": [
                {
                    "scene_index": index,
                    "heading": scene.heading,
                    "body": scene.body,
                    "visual": scene.visual,
                    "source_index": scene.source_index,
                }
                for index, scene in enumerate(package.scenes)
            ],
            "sources": _source_context(package, sources),
            "strategy": {
                "hook": strategy.hook,
                "pacing": strategy.pacing,
                "visual": strategy.visual,
                "duration": strategy.duration,
            },
        },
        "working_dir": str((output_dir / "vimax-workspace").resolve()),
    }


def run_vimax_planner(
    *,
    settings: Settings,
    package: VideoPackage,
    sources: Sequence[SourceItem],
    strategy: Strategy,
    output_dir: Path,
) -> ViMaxPlan:
    output_dir.mkdir(parents=True, exist_ok=True)
    request = _request_payload(
        settings=settings,
        package=package,
        sources=sources,
        strategy=strategy,
        output_dir=output_dir,
    )
    request_path = output_dir / "vimax-planner-request.json"
    response_path = output_dir / "vimax-plan.json"
    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")

    command = [
        _planner_python(),
        str(_planner_script()),
        "--vimax-root",
        str(_vimax_root()),
        "--request",
        str(request_path),
        "--output",
        str(response_path),
    ]
    timeout = int(os.getenv("VIMAX_PLANNER_TIMEOUT_SECONDS", "900"))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    (output_dir / "vimax-planner.log").write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ViMaxPlanningError(
            f"ViMax planner exited with {completed.returncode}: {completed.stderr[-4000:]}"
        )
    if not response_path.is_file():
        raise ViMaxPlanningError("ViMax planner did not write vimax-plan.json")
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ViMaxPlanningError(f"ViMax planner returned invalid JSON: {exc}") from exc
    if payload.get("status") != "planned":
        raise ViMaxPlanningError(f"ViMax planner did not pass: {payload}")
    if payload.get("vimax_commit") != VIMAX_COMMIT:
        raise ViMaxPlanningError("ViMax planner commit does not match the pinned contract")
    raw_shots = payload.get("shot_descriptions")
    if not isinstance(raw_shots, list):
        raise ViMaxPlanningError("ViMax plan is missing shot_descriptions")
    profile = VideoProfile.from_env()
    if not profile.minimum_shots <= len(raw_shots) <= profile.maximum_shots:
        raise ViMaxPlanningError(
            f"ViMax planned {len(raw_shots)} shots; required "
            f"{profile.minimum_shots}-{profile.maximum_shots}"
        )
    shots = tuple(
        ViMaxShot.from_dict(item, expected_idx=index)
        for index, item in enumerate(raw_shots)
        if isinstance(item, dict)
    )
    if len(shots) != len(raw_shots):
        raise ViMaxPlanningError("every ViMax shot must be an object")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ViMaxPlan(
        shots=shots,
        characters=tuple(item for item in payload.get("characters", []) if isinstance(item, dict)),
        camera_tree=tuple(item for item in payload.get("camera_tree", []) if isinstance(item, dict)),
        artifact_path=response_path,
        payload_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def _source_index_for_shot(index: int, shot_count: int, package: VideoPackage) -> int:
    package_index = min(len(package.scenes) - 1, index * len(package.scenes) // shot_count)
    return int(package.scenes[package_index].source_index)


def _role_for_shot(index: int, count: int) -> str:
    if index == 0:
        return "hook"
    if index == count - 1:
        return "cta"
    ratio = index / max(1, count - 1)
    if ratio < 0.32:
        return "evidence"
    if ratio < 0.58:
        return "mechanism"
    if ratio < 0.80:
        return "comparison"
    return "implication"


def _wan_indices(shots: Sequence[ViMaxShot], profile: VideoProfile) -> set[int]:
    ranked = sorted(
        range(1, len(shots)),
        key=lambda index: (
            {"large": 3, "medium": 2, "small": 1}[shots[index].variation_type],
            len(shots[index].motion_prompt.split()),
            -index,
        ),
        reverse=True,
    )
    selected = {0, *ranked[: max(0, profile.wan_shots - 1)]}
    if len(selected) != profile.wan_shots:
        raise ViMaxPlanningError("ViMax plan cannot satisfy the exact Wan shot budget")
    return selected


def construct_vimax_visual_plan(
    settings: Settings,
    package: VideoPackage,
    sources: list[SourceItem],
    strategy: Strategy,
    *,
    plan_validator: Callable[[VisualPlan], None] | None = None,
) -> VisualPlan:
    """Use ViMax Script2Video for planning while preserving factory media/QC contracts."""
    work_root = Path(settings.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    planning_root = Path(tempfile.mkdtemp(prefix="vimax-plan-", dir=work_root))
    result = run_vimax_planner(
        settings=settings,
        package=package,
        sources=sources,
        strategy=strategy,
        output_dir=planning_root,
    )
    profile = VideoProfile.from_env()
    wan = _wan_indices(result.shots, profile)
    scenes: list[SceneVisualPrompt] = []
    for index, shot in enumerate(result.shots):
        source_index = _source_index_for_shot(index, len(result.shots), package)
        scenes.append(
            SceneVisualPrompt(
                scene_index=index,
                source_index=source_index,
                role=_role_for_shot(index, len(result.shots)),
                generation_mode="wan_i2v" if index in wan else "image",
                image_prompt=(
                    shot.first_frame_prompt.rstrip(" .")
                    + ". Vertical 9:16 photorealistic technology documentary frame; "
                    "one continuous camera view, realistic materials and anatomy, no readable "
                    "text, logos, watermarks, interfaces, split panels, or collage."
                ),
                motion_prompt=(
                    shot.motion_prompt.rstrip(" .")
                    + ". Preserve the first-frame subject identity, environment, geometry, and "
                    "camera continuity; no cut, shake, morphing, new object, or disappearing object."
                ),
                negative_prompt=(
                    "readable text, pseudo-text, letters, numbers, logos, trademarks, watermark, "
                    "signature, UI, collage, split frame, malformed anatomy, warped equipment, "
                    "duplicate subject, camera shake, flicker, morphing, abrupt cut"
                ),
                continuity_anchor=(
                    f"ViMax camera {shot.cam_idx}; {shot.variation_type} variation; "
                    f"last frame: {shot.last_frame_prompt}"
                ),
                caption_safe_zone="lower_20_percent_overlay_only",
                seed=(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{package.title}|{index}|{VIMAX_PROMPT_VERSION}".encode()
                        ).digest()[:4],
                        "big",
                    )
                    & 0x7FFFFFFF
                ),
                duration_seconds=round(settings.target_seconds / len(result.shots), 3),
            )
        )
    plan = VisualPlan(
        prompt_version=VIMAX_PROMPT_VERSION,
        global_style=(
            "ViMax-directed photorealistic vertical technology documentary with coherent "
            "cinematography and source-grounded physical storytelling"
        ),
        palette="neutral graphite materials with restrained blue and amber practical accents",
        lighting="natural documentary lighting with consistent exposure and cinematic depth",
        continuity_bible=(
            f"ViMax planning artifact {result.artifact_path}; sha256={result.payload_sha256}; "
            "preserve recurring subjects, camera relationships, environments, and physical props"
        ),
        image_model=os.getenv("VISUAL_FLUX_MODEL", "black-forest-labs/FLUX.1-schnell"),
        video_model=os.getenv("WAN22_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        width=704,
        height=1280,
        fps=24,
        director_input_sha256=result.payload_sha256,
        scenes=tuple(scenes),
    )
    _PLAN_ARTIFACTS[plan.director_input_sha256] = result.artifact_path
    if plan_validator is not None:
        plan_validator(plan)
    return plan


def persist_vimax_plan_artifact(plan: VisualPlan, visual_root: Path) -> Path | None:
    """Copy the exact ViMax planner output beside the canonical visual plan."""
    if not plan.prompt_version.startswith("vimax-script2video@"):
        return None
    source = _PLAN_ARTIFACTS.get(plan.director_input_sha256)
    if source is None or not source.is_file():
        raise ViMaxPlanningError("the exact ViMax planning artifact is unavailable")
    destination = visual_root / "vimax-plan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    shutil.rmtree(source.parent, ignore_errors=True)
    _PLAN_ARTIFACTS.pop(plan.director_input_sha256, None)
    return destination


def _caption_claims(segments: Sequence[NarrationSegment]) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    for segment in sorted(segments, key=lambda item: item.segment_id):
        text = " ".join(segment.text.split()).strip()
        if text:
            values.append((int(segment.segment_id), text))
    if not values:
        raise ViMaxPlanningError("reviewed narration produced no claims")
    return values


def _balanced_frames(total_frames: int, shot_count: int, profile: VideoProfile, fps: int) -> list[int]:
    minimum = max(1, round(profile.minimum_shot_seconds * fps))
    maximum = max(minimum, round(profile.maximum_shot_seconds * fps))
    if total_frames < shot_count * minimum or total_frames > shot_count * maximum:
        raise ViMaxPlanningError(
            f"{shot_count} ViMax shots cannot cover {total_frames / fps:.3f}s inside "
            f"{profile.minimum_shot_seconds}-{profile.maximum_shot_seconds}s bounds"
        )
    base, remainder = divmod(total_frames, shot_count)
    frames = [base + (1 if index < remainder else 0) for index in range(shot_count)]
    if any(value < minimum or value > maximum for value in frames):
        raise ViMaxPlanningError("balanced ViMax durations violate shot bounds")

    # The fourth shot must begin before ten seconds. Move only the necessary frames
    # from the first three shots into later shots while keeping all bounds exact.
    if shot_count >= 4:
        opening_limit = max(0, round(10.0 * fps) - 1)
        opening_frames = sum(frames[:3])
        excess = max(0, opening_frames - opening_limit)
        if excess:
            donors = sorted(range(3), key=lambda index: frames[index], reverse=True)
            recipients = list(range(3, shot_count))
            for donor in donors:
                removable = min(excess, frames[donor] - minimum)
                if removable <= 0:
                    continue
                remaining = removable
                for recipient in recipients:
                    room = maximum - frames[recipient]
                    moved = min(room, remaining)
                    frames[recipient] += moved
                    frames[donor] -= moved
                    remaining -= moved
                    excess -= moved
                    if remaining == 0:
                        break
                if excess == 0:
                    break
            if excess:
                raise ViMaxPlanningError(
                    "ViMax durations cannot satisfy four opening shots inside ten seconds"
                )
    return frames


def build_vimax_editorial_plan(
    *,
    plan: VisualPlan,
    package: VideoPackage,
    segments: Sequence[NarrationSegment],
    total_duration: float,
    profile: VideoProfile,
) -> tuple[VisualPlan, tuple[Any, ...]]:
    """Convert an already decomposed ViMax storyboard into the factory ShotSpec contract."""
    from .editorial_timeline import ShotSpec

    if not plan.prompt_version.startswith("vimax-script2video@"):
        raise ViMaxPlanningError("build_vimax_editorial_plan received a non-ViMax plan")
    shot_count = len(plan.scenes)
    if not profile.minimum_shots <= shot_count <= profile.maximum_shots:
        raise ViMaxPlanningError("ViMax shot count is outside the production profile")
    fps = 30
    total_frames = round(total_duration * fps)
    frames = _balanced_frames(total_frames, shot_count, profile, fps)
    wan_indices = {index for index, scene in enumerate(plan.scenes) if scene.generation_mode == "wan_i2v"}
    if len(wan_indices) != profile.wan_shots:
        raise ViMaxPlanningError("ViMax plan changed the exact Wan shot budget")
    wan_maximum = round(profile.maximum_wan_shot_seconds * fps)
    # Move excess Wan time into non-Wan shots without changing the total frame count.
    for index in sorted(wan_indices):
        excess = max(0, frames[index] - wan_maximum)
        if not excess:
            continue
        recipients = [
            candidate
            for candidate in range(shot_count)
            if candidate not in wan_indices
            and frames[candidate] < round(profile.maximum_shot_seconds * fps)
        ]
        for candidate in recipients:
            room = round(profile.maximum_shot_seconds * fps) - frames[candidate]
            moved = min(room, excess)
            frames[candidate] += moved
            frames[index] -= moved
            excess -= moved
            if excess == 0:
                break
        if excess:
            raise ViMaxPlanningError("ViMax Wan duration ceiling cannot be satisfied")
    claims = _caption_claims(segments)
    cursor_frames = 0
    shots: list[ShotSpec] = []
    updated_scenes: list[SceneVisualPrompt] = []
    for index, (scene, duration_frames) in enumerate(zip(plan.scenes, frames, strict=True)):
        segment_id, claim = claims[min(len(claims) - 1, index * len(claims) // shot_count)]
        package_scene_index = min(len(package.scenes) - 1, index * len(package.scenes) // shot_count)
        duration_seconds = duration_frames / fps
        shots.append(
            ShotSpec(
                shot_id=index,
                beat_index=index,
                segment_id=segment_id,
                package_scene_index=package_scene_index,
                source_index=scene.source_index,
                start_seconds=round(cursor_frames / fps, 6),
                duration_seconds=round(duration_seconds, 6),
                renderer="wan_i2v" if scene.generation_mode == "wan_i2v" else "parallax",
                semantic_claim=claim,
                visual_direction=scene.image_prompt,
                treatment=scene.continuity_anchor,
                seed=scene.seed,
            )
        )
        updated_scenes.append(replace(scene, duration_seconds=round(duration_seconds, 6)))
        cursor_frames += duration_frames
    if cursor_frames != total_frames:
        raise ViMaxPlanningError("ViMax timeline does not cover the reviewed narration")
    if sum(shot.start_seconds < 10.0 for shot in shots) < profile.first_ten_seconds_minimum_shots:
        raise ViMaxPlanningError("ViMax opening does not contain four shots in ten seconds")
    return replace(plan, scenes=tuple(updated_scenes)), tuple(shots)
