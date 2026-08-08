from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence


_INSTALLED = False
_STATIC_MOTION_RE = re.compile(
    r"\b(?:static camera|locked camera|no significant changes?|no changes?|"
    r"no movement|no camera movement|unchanged composition|remains in the same position)\b",
    re.IGNORECASE,
)
_MOTION_CYCLE = (
    "Slow dolly in while the primary subject completes the depicted action; visible hand, tool, device, or environment motion must progress from the first frame to the last frame.",
    "Controlled track right following the primary subject through the depicted workflow; preserve identity and geometry while the action visibly advances across the shot.",
    "Slow dolly out revealing the relationship between the primary subject and the surrounding workflow; require continuous subject or environmental movement rather than a frozen pose.",
    "Controlled pan left following the depicted action from cause to result; preserve all subjects and props while their state visibly changes over time.",
    "Gentle tilt down following the primary subject into the concrete task; require visible temporal change in the subject, tool, device, or environment throughout the shot.",
    "Controlled push in toward the consequence of the depicted action; preserve continuity while the physical workflow visibly progresses between first and last frame.",
)


class ViMaxTemporalVideoError(RuntimeError):
    """Fail closed when the ViMax path degrades into still-image animation."""


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def temporal_profile_v55(profile: Any) -> Any:
    """Make the ViMax release contract exactly target_shots native temporal clips."""
    maximum_temporal = min(
        float(profile.maximum_shot_seconds),
        float(profile.maximum_wan_shot_seconds),
    )
    updated = replace(
        profile,
        maximum_shot_seconds=maximum_temporal,
        wan_shots=int(profile.target_shots),
    )
    updated.validate()
    return updated


def _repair_motion_prompt(value: str, index: int) -> str:
    clean = " ".join(str(value or "").split()).strip(" .")
    if not clean or _STATIC_MOTION_RE.search(clean):
        clean = _MOTION_CYCLE[index % len(_MOTION_CYCLE)].strip(" .")
    return (
        clean
        + ". Native temporal-generation requirement: the source clip itself must contain "
        "visible subject, object, or environmental motion from first frame to last frame; "
        "do not simulate motion with a crop, digital zoom, frozen hold, or slideshow transform."
    )


def ensure_temporal_vimax_plan_v55(plan: Any) -> Any:
    """Convert every ViMax shot into a real image-to-video generation request."""
    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    scenes = tuple(
        replace(
            scene,
            generation_mode="wan_i2v",
            motion_prompt=_repair_motion_prompt(scene.motion_prompt, index),
        )
        for index, scene in enumerate(plan.scenes)
    )
    if len(scenes) != len(plan.scenes):
        raise ViMaxTemporalVideoError("ViMax temporal conversion changed the shot count")
    return replace(plan, scenes=scenes)


def frame_num_for_duration_v55(
    duration_seconds: float,
    fps: int,
    *,
    minimum: int = 41,
    maximum: int = 81,
) -> int:
    """Return the smallest Wan-compatible 4n+1 frame count covering the shot duration."""
    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration_seconds and fps must be positive")
    required = max(minimum, int(math.ceil(duration_seconds * fps)))
    candidate = required
    remainder = (candidate - 1) % 4
    if remainder:
        candidate += 4 - remainder
    if candidate > maximum:
        raise ViMaxTemporalVideoError(
            f"shot requires {candidate} Wan frames, above configured maximum {maximum}"
        )
    if candidate < 17 or (candidate - 1) % 4 != 0:
        raise ViMaxTemporalVideoError("computed Wan frame count violates the 4n+1 contract")
    return candidate


def native_temporal_media_failures_v55(media_types: Sequence[str]) -> tuple[str, ...]:
    failures: list[str] = []
    for index, media_type in enumerate(media_types):
        if str(media_type).strip().lower() != "video":
            failures.append(
                f"ViMax shot {index} is not native temporal video media; slideshow/image fallback is forbidden"
            )
    return tuple(failures)


def _install_profile_contract() -> None:
    from .video_profile import VideoProfile

    current = VideoProfile.from_env.__func__
    if getattr(current, "_agf_v55", False):
        return

    def from_env_v55(cls: type[Any]) -> Any:
        profile = current(cls)
        if _enabled():
            profile = temporal_profile_v55(profile)
        return profile

    from_env_v55._agf_v55 = True  # type: ignore[attr-defined]
    VideoProfile.from_env = classmethod(from_env_v55)


def _install_vimax_planner_contract() -> None:
    from . import vimax_planner

    current_run = vimax_planner.run_vimax_planner
    if not getattr(current_run, "_agf_v55", False):
        def run_vimax_planner_v55(*args: Any, **kwargs: Any) -> Any:
            result = current_run(*args, **kwargs)
            from .video_profile import VideoProfile

            profile = VideoProfile.from_env()
            if len(result.shots) != profile.target_shots:
                raise ViMaxTemporalVideoError(
                    f"ViMax returned {len(result.shots)} shots; temporal release contract requires "
                    f"exactly {profile.target_shots}"
                )
            return result

        run_vimax_planner_v55._agf_v55 = True  # type: ignore[attr-defined]
        vimax_planner.run_vimax_planner = run_vimax_planner_v55

    def all_temporal_indices(shots: Sequence[Any], profile: Any) -> set[int]:
        if len(shots) != int(profile.target_shots):
            raise ViMaxTemporalVideoError(
                f"cannot allocate temporal media for {len(shots)} shots; expected {profile.target_shots}"
            )
        return set(range(len(shots)))

    vimax_planner._wan_indices = all_temporal_indices


def _install_vimax_authority_bridge() -> None:
    from . import production_vimax_visual_authority_v52 as authority_v52

    current = authority_v52._enrich_from_vimax_artifact
    if getattr(current, "_agf_v55", False):
        return

    def enrich_v55(plan: Any, package: Any) -> Any:
        return ensure_temporal_vimax_plan_v55(current(plan, package))

    enrich_v55._agf_v55 = True  # type: ignore[attr-defined]
    authority_v52._enrich_from_vimax_artifact = enrich_v55


def _load_wan_without_cpu_offload(self: Any) -> Any:
    if self._pipeline is not None:
        return self._pipeline
    try:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    except ImportError as exc:
        raise ViMaxTemporalVideoError("Wan2.2 Diffusers dependencies are missing") from exc
    token = os.getenv("HF_TOKEN") or None
    try:
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            token=token,
        )
        pipeline = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        if not torch.cuda.is_available():
            raise ViMaxTemporalVideoError("native temporal ViMax generation requires CUDA")
        pipeline.to("cuda")
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        pipeline.set_progress_bar_config(disable=True)
    except Exception as exc:
        if isinstance(exc, ViMaxTemporalVideoError):
            raise
        raise ViMaxTemporalVideoError(f"Could not load {self.model_id} on GPU: {exc}") from exc
    self._pipeline = pipeline
    return pipeline


def _install_dynamic_wan_generation() -> None:
    from . import video_generator

    current_load = video_generator.Wan22DiffusersAnimator._load
    if not getattr(current_load, "_agf_v55", False):
        def load_v55(self: Any) -> Any:
            use_cpu_offload = os.getenv("WAN22_MODEL_CPU_OFFLOAD", "true").strip().casefold() in {
                "1", "true", "yes", "on"
            }
            if use_cpu_offload:
                return current_load(self)
            return _load_wan_without_cpu_offload(self)

        load_v55._agf_v55 = True  # type: ignore[attr-defined]
        video_generator.Wan22DiffusersAnimator._load = load_v55

    current_animate = video_generator.Wan22DiffusersAnimator.animate
    if not getattr(current_animate, "_agf_v55", False):
        def animate_v55(self: Any, scene: Any, keyframe: Any, output: Path) -> Any:
            minimum = int(os.getenv("WAN22_MIN_FRAME_NUM", "41"))
            maximum = int(os.getenv("WAN22_MAX_FRAME_NUM", "81"))
            dynamic = os.getenv("WAN22_DYNAMIC_FRAME_NUM", "true").strip().casefold() in {
                "1", "true", "yes", "on"
            }
            original = int(self.frame_num)
            selected = original
            if _enabled() and dynamic:
                selected = frame_num_for_duration_v55(
                    float(scene.duration_seconds),
                    int(self.plan.fps),
                    minimum=minimum,
                    maximum=maximum,
                )
            self.frame_num = selected
            try:
                return current_animate(self, scene, keyframe, output)
            finally:
                self.frame_num = original

        animate_v55._agf_v55 = True  # type: ignore[attr-defined]
        video_generator.Wan22DiffusersAnimator.animate = animate_v55

    current_generate = video_generator.generate_scene_media
    if getattr(current_generate, "_agf_v55", False):
        return

    def generate_scene_media_v55(plan: Any, keyframes: Any, output_dir: Path) -> Any:
        assets = current_generate(plan, keyframes, output_dir)
        if not _enabled() or not str(plan.prompt_version).startswith("vimax-script2video@"):
            return assets
        media_types = tuple(str(asset.media_type).strip().lower() for asset in assets)
        failures = native_temporal_media_failures_v55(media_types)
        if failures:
            raise ViMaxTemporalVideoError("; ".join(failures))
        if len(assets) != len(plan.scenes):
            raise ViMaxTemporalVideoError("ViMax temporal media count does not match the plan")
        if len({asset.sha256 for asset in assets}) != len(assets):
            raise ViMaxTemporalVideoError("ViMax temporal generation reused a source clip")
        for asset in assets:
            if not asset.path.is_file() or asset.path.suffix.lower() != ".mp4":
                raise ViMaxTemporalVideoError(
                    f"ViMax shot {asset.scene_index} did not produce an MP4 source clip"
                )

        manifest_path = output_dir / "scene-media-manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        minimum = int(os.getenv("WAN22_MIN_FRAME_NUM", "41"))
        maximum = int(os.getenv("WAN22_MAX_FRAME_NUM", "81"))
        payload.update(
            {
                "native_temporal_source_required": True,
                "image_fallback_allowed": False,
                "digital_zoom_motion_allowed": False,
                "expected_temporal_shots": len(plan.scenes),
                "realized_temporal_shots": len(assets),
                "dynamic_frame_num": True,
                "per_scene_frame_num": {
                    str(scene.scene_index): frame_num_for_duration_v55(
                        float(scene.duration_seconds),
                        int(plan.fps),
                        minimum=minimum,
                        maximum=maximum,
                    )
                    for scene in plan.scenes
                },
            }
        )
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return assets

    generate_scene_media_v55._agf_v55 = True  # type: ignore[attr-defined]
    video_generator.generate_scene_media = generate_scene_media_v55


def _install_temporal_qc() -> None:
    from . import production_video_qc

    current = production_video_qc.production_motion_failures
    if getattr(current, "_agf_v55", False):
        return

    def temporal_motion_failures_v55(media_types: Sequence[str], report: Any) -> tuple[str, ...]:
        failures = list(current(media_types, report))
        if _enabled():
            failures.extend(native_temporal_media_failures_v55(media_types))
        return tuple(dict.fromkeys(failures))

    temporal_motion_failures_v55._agf_v55 = True  # type: ignore[attr-defined]
    production_video_qc.production_motion_failures = temporal_motion_failures_v55


def install_production_vimax_temporal_video_v55() -> None:
    """Require real generated video for every ViMax shot and make Remotion edit those clips."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    _install_profile_contract()
    _install_vimax_planner_contract()
    _install_vimax_authority_bridge()
    _install_dynamic_wan_generation()
    _install_temporal_qc()
    _INSTALLED = True
