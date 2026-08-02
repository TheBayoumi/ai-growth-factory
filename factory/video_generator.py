from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .image_generator import KeyframeAsset
from .visual_prompt import SceneVisualPrompt, VisualPlan


class VideoGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneMediaAsset:
    scene_index: int
    media_type: str
    path: Path
    keyframe_path: Path
    model: str
    seed: int
    prompt: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["keyframe_path"] = str(self.keyframe_path)
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keyframe_by_scene(keyframes: tuple[KeyframeAsset, ...]) -> dict[int, KeyframeAsset]:
    result = {asset.scene_index: asset for asset in keyframes}
    if len(result) != len(keyframes):
        raise VideoGenerationError("Duplicate keyframe scene index")
    return result


class Wan22DiffusersAnimator:
    """Generate hero clips with the official Wan2.2 TI2V-5B Diffusers checkpoint."""

    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.model_id = os.getenv(
            "WAN22_MODEL_ID",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ).strip()
        self.steps = int(os.getenv("WAN22_SAMPLE_STEPS", "30"))
        self.frame_num = int(os.getenv("WAN22_FRAME_NUM", "81"))
        self.guidance_scale = float(os.getenv("WAN22_GUIDANCE_SCALE", "5.0"))
        if self.frame_num < 17 or (self.frame_num - 1) % 4 != 0:
            raise VideoGenerationError("WAN22_FRAME_NUM must be 4n+1 and at least 17")
        if not 8 <= self.steps <= 60:
            raise VideoGenerationError("WAN22_SAMPLE_STEPS must be between 8 and 60")
        if not 1.0 <= self.guidance_scale <= 12.0:
            raise VideoGenerationError("WAN22_GUIDANCE_SCALE must be between 1 and 12")
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        except ImportError as exc:
            raise VideoGenerationError(
                "Wan2.2 Diffusers dependencies are missing from the visual worker"
            ) from exc
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
            pipeline.enable_model_cpu_offload()
            if hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
            if hasattr(pipeline.vae, "enable_slicing"):
                pipeline.vae.enable_slicing()
            pipeline.set_progress_bar_config(disable=True)
        except Exception as exc:
            raise VideoGenerationError(f"Could not load {self.model_id}: {exc}") from exc
        self._pipeline = pipeline
        return pipeline

    def animate(
        self,
        scene: SceneVisualPrompt,
        keyframe: KeyframeAsset,
        output: Path,
    ) -> Path:
        pipeline = self._load()
        try:
            import torch
            from diffusers.utils import export_to_video
        except ImportError as exc:
            raise VideoGenerationError("Wan2.2 export dependencies are missing") from exc

        with Image.open(keyframe.path) as source:
            image = source.convert("RGB").resize(
                (self.plan.width, self.plan.height),
                Image.Resampling.LANCZOS,
            )
        generator = torch.Generator(device="cpu").manual_seed(scene.seed)
        prompt = (
            scene.motion_prompt
            + " Maintain this source-grounded visual design: "
            + scene.image_prompt
        )
        try:
            frames = pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=scene.negative_prompt,
                height=self.plan.height,
                width=self.plan.width,
                num_frames=self.frame_num,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).frames[0]
            output.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output), fps=self.plan.fps)
        except Exception as exc:
            raise VideoGenerationError(
                f"Wan2.2 failed for scene {scene.scene_index}: {exc}"
            ) from exc
        if not output.is_file() or output.stat().st_size < 250_000:
            raise VideoGenerationError(
                f"Wan2.2 produced no usable clip for scene {scene.scene_index}"
            )
        return output


def generate_scene_media(
    plan: VisualPlan,
    keyframes: tuple[KeyframeAsset, ...],
    output_dir: Path,
) -> tuple[SceneMediaAsset, ...]:
    """Generate Wan clips for hero scenes and preserve keyframes for supporting scenes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_map = _keyframe_by_scene(keyframes)
    animator = Wan22DiffusersAnimator(plan)
    assets: list[SceneMediaAsset] = []

    for scene in plan.scenes:
        keyframe = keyframe_map.get(scene.scene_index)
        if keyframe is None:
            raise VideoGenerationError(f"Missing keyframe for scene {scene.scene_index}")
        if scene.generation_mode == "wan_i2v":
            path = output_dir / f"scene-{scene.scene_index:02d}-wan.mp4"
            animator.animate(scene, keyframe, path)
            media_type = "video"
            model = animator.model_id
            prompt = scene.motion_prompt
        elif scene.generation_mode == "image":
            path = keyframe.path
            media_type = "image"
            model = keyframe.model
            prompt = scene.image_prompt
        else:
            raise VideoGenerationError(
                f"Unsupported generation mode: {scene.generation_mode}"
            )
        assets.append(
            SceneMediaAsset(
                scene_index=scene.scene_index,
                media_type=media_type,
                path=path,
                keyframe_path=keyframe.path,
                model=model,
                seed=scene.seed,
                prompt=prompt,
                sha256=_sha256(path),
            )
        )

    if sum(asset.media_type == "video" for asset in assets) != 3:
        raise VideoGenerationError("Production visual plan must contain exactly three Wan clips")

    manifest = {
        "video_backend": "wan22_ti2v_diffusers",
        "video_model": animator.model_id,
        "frame_num": animator.frame_num,
        "sample_steps": animator.steps,
        "guidance_scale": animator.guidance_scale,
        "model_cpu_offload": True,
        "vae_tiling": True,
        "vae_slicing": True,
        "assets": [asset.as_dict() for asset in assets],
    }
    (output_dir / "scene-media-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tuple(assets)
