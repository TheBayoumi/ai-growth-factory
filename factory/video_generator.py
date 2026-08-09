from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .image_generator import KeyframeAsset
from .visual_prompt import SceneVisualPrompt, VisualPlan
from .visual_prompt_compiler import compile_motion_prompt


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
    director_prompt: str = ""
    prompt_word_count: int = 0
    prompt_word_budget: int = 0
    prompt_compiler_version: str = ""

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


def _validate_wan_media_budget(
    plan: VisualPlan,
    assets: Iterable[SceneMediaAsset],
) -> tuple[int, int]:
    """Require realized Wan clips to match the plan-derived editorial budget."""
    expected = sum(scene.generation_mode == "wan_i2v" for scene in plan.scenes)
    actual = sum(asset.media_type == "video" for asset in assets)
    if expected < 1:
        raise VideoGenerationError("Production visual plan must contain at least one Wan clip")
    if actual != expected:
        raise VideoGenerationError(
            f"Production visual plan requires {expected} Wan clips; generated {actual}"
        )
    return expected, actual


def _frame_to_uint8(frame: Any) -> Any:
    """Normalize PIL, NumPy, or Torch Wan frames to contiguous RGB uint8 arrays.

    Diffusers can return float32 HWC frames in [0, 1], float frames in [-1, 1],
    or channel-first tensors. Pillow cannot construct RGB images directly from a
    three-channel float array, which caused the v8 production failure after Wan
    inference had already completed.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise VideoGenerationError("Wan frame normalization requires numpy") from exc

    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    value = frame
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()

    array = np.asarray(value)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise VideoGenerationError(
            f"Wan frame must resolve to HxWxC; received shape {array.shape}"
        )
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise VideoGenerationError(
            f"Wan frame must contain 1, 3, or 4 channels; received shape {array.shape}"
        )

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum >= 0.0 and maximum <= 1.5:
            array = array * 255.0
        elif minimum >= -1.05 and maximum <= 1.05:
            array = (array + 1.0) * 127.5
        array = np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _export_frames(frames: Iterable[Any], output: Path, *, fps: int) -> Path:
    """Export normalized Wan frames through the pinned imageio FFmpeg backend."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise VideoGenerationError(
            "Wan frame export requires imageio, imageio-ffmpeg, and numpy"
        ) from exc
    arrays = [_frame_to_uint8(frame) for frame in frames]
    if not arrays:
        raise VideoGenerationError("Wan returned no frames to export")
    expected_shape = arrays[0].shape
    if any(array.shape != expected_shape for array in arrays):
        raise VideoGenerationError("Wan returned frames with inconsistent dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimwrite(
            str(output),
            arrays,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=None,
        )
    except Exception as exc:
        raise VideoGenerationError(f"imageio could not export Wan frames: {exc}") from exc
    return output


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
            raise VideoGenerationError("WAN22_GUIDANCE_SCALE must be between 1.0 and 12.0")
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
    ) -> tuple[Path, str, int, int, str]:
        pipeline = self._load()
        try:
            import torch
        except ImportError as exc:
            raise VideoGenerationError("PyTorch is missing from the Wan worker") from exc

        with Image.open(keyframe.path) as source:
            image = source.convert("RGB").resize(
                (self.plan.width, self.plan.height),
                Image.Resampling.LANCZOS,
            )
        generator = torch.Generator(device="cpu").manual_seed(scene.seed)
        executable = compile_motion_prompt(scene.motion_prompt)
        try:
            frames = pipeline(
                image=image,
                prompt=executable.compiled_motion_prompt,
                negative_prompt=(
                    "text, letters, numbers, symbols, captions, logos, watermark, screens, "
                    "new objects, cuts, camera shake, zoom, flicker, morphing, warped anatomy"
                ),
                height=self.plan.height,
                width=self.plan.width,
                num_frames=self.frame_num,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).frames[0]
            _export_frames(frames, output, fps=self.plan.fps)
        except Exception as exc:
            if isinstance(exc, VideoGenerationError):
                raise
            raise VideoGenerationError(
                f"Wan2.2 failed for scene {scene.scene_index}: {exc}"
            ) from exc
        if not output.is_file() or output.stat().st_size < 250_000:
            raise VideoGenerationError(
                f"Wan2.2 produced no usable clip for scene {scene.scene_index}"
            )
        return (
            output,
            executable.compiled_motion_prompt,
            executable.word_count,
            executable.word_budget,
            executable.compiler_version,
        )


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
            (
                path,
                compiled_prompt,
                prompt_word_count,
                prompt_word_budget,
                compiler_version,
            ) = animator.animate(scene, keyframe, path)
            media_type = "video"
            model = animator.model_id
            prompt = compiled_prompt
            director_prompt = scene.motion_prompt
        elif scene.generation_mode == "image":
            path = keyframe.path
            media_type = "image"
            model = keyframe.model
            prompt = keyframe.prompt
            director_prompt = keyframe.director_prompt
            prompt_word_count = keyframe.prompt_word_count
            prompt_word_budget = keyframe.prompt_word_budget
            compiler_version = keyframe.prompt_compiler_version
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
                director_prompt=director_prompt,
                prompt_word_count=prompt_word_count,
                prompt_word_budget=prompt_word_budget,
                prompt_compiler_version=compiler_version,
            )
        )

    expected_wan_shots, realized_wan_shots = _validate_wan_media_budget(plan, assets)

    manifest = {
        "video_backend": "wan22_ti2v_diffusers_imageio_export",
        "video_model": animator.model_id,
        "frame_num": animator.frame_num,
        "sample_steps": animator.steps,
        "guidance_scale": animator.guidance_scale,
        "model_cpu_offload": True,
        "vae_tiling": True,
        "vae_slicing": True,
        "frame_normalization": "float_or_tensor_to_contiguous_rgb_uint8",
        "image_prompt_reinjected_into_motion_prompt": False,
        "expected_wan_shots": expected_wan_shots,
        "realized_wan_shots": realized_wan_shots,
        "assets": [asset.as_dict() for asset in assets],
    }
    (output_dir / "scene-media-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tuple(assets)
