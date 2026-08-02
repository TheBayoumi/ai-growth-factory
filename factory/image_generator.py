from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageStat

from .visual_prompt import VisualPlan


class ImageGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyframeAsset:
    scene_index: int
    path: Path
    model: str
    seed: int
    width: int
    height: int
    sha256: str
    entropy: float
    prompt: str
    negative_prompt: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(image: Image.Image, size: int = 16) -> int:
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _validate_keyframe(path: Path, *, width: int, height: int) -> tuple[float, int]:
    if not path.is_file() or path.stat().st_size < 80_000:
        raise ImageGenerationError(f"Generated keyframe is missing or too small: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise ImageGenerationError(
                f"Generated keyframe has wrong size {image.size}; expected {(width, height)}"
            )
        entropy = float(image.convert("RGB").entropy())
        if entropy < 4.2:
            raise ImageGenerationError(
                f"Generated keyframe is visually under-detailed: entropy={entropy:.3f}"
            )
        extrema = ImageStat.Stat(image.convert("RGB")).extrema
        if any(high - low < 35 for low, high in extrema):
            raise ImageGenerationError("Generated keyframe has insufficient tonal range")
        image_hash = _average_hash(image)
    return entropy, image_hash


def _materialize_assets(
    *,
    plan: VisualPlan,
    output_dir: Path,
    backend: str,
    model: str,
    steps: int,
    infer: Callable[[str, str, int], Image.Image],
) -> tuple[KeyframeAsset, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[KeyframeAsset] = []
    hashes: list[int] = []
    for scene in plan.scenes:
        try:
            image = infer(scene.image_prompt, scene.negative_prompt, scene.seed).convert("RGB")
        except Exception as exc:
            raise ImageGenerationError(
                f"{backend} failed for scene {scene.scene_index}: {exc}"
            ) from exc
        if image.size != (plan.width, plan.height):
            image = image.resize((plan.width, plan.height), Image.Resampling.LANCZOS)
        path = output_dir / f"scene-{scene.scene_index:02d}-keyframe.png"
        image.save(path, format="PNG", optimize=True)
        entropy, image_hash = _validate_keyframe(path, width=plan.width, height=plan.height)
        if any(_hamming(image_hash, previous) < 18 for previous in hashes):
            raise ImageGenerationError(
                f"Scene {scene.scene_index} keyframe is too similar to another scene"
            )
        hashes.append(image_hash)
        assets.append(
            KeyframeAsset(
                scene_index=scene.scene_index,
                path=path,
                model=model,
                seed=scene.seed,
                width=plan.width,
                height=plan.height,
                sha256=_sha256(path),
                entropy=round(entropy, 4),
                prompt=scene.image_prompt,
                negative_prompt=scene.negative_prompt,
            )
        )

    manifest = {
        "backend": backend,
        "model": model,
        "steps": steps,
        "width": plan.width,
        "height": plan.height,
        "captions_or_text_requested": False,
        "assets": [asset.as_dict() for asset in assets],
    }
    (output_dir / "keyframe-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tuple(assets)


class FluxKeyframeGenerator:
    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.model_id = os.getenv(
            "VISUAL_FLUX_MODEL", "black-forest-labs/FLUX.1-schnell"
        ).strip()
        self.steps = int(os.getenv("VISUAL_FLUX_STEPS", "4"))
        if not 1 <= self.steps <= 12:
            raise ImageGenerationError("VISUAL_FLUX_STEPS must be between 1 and 12")
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImageGenerationError("FLUX dependencies are missing") from exc
        token = os.getenv("HF_TOKEN") or None
        if not token:
            raise ImageGenerationError(
                "FLUX was selected but HF_TOKEN is missing; use VISUAL_IMAGE_BACKEND=auto "
                "for the public SDXL-Lightning fallback"
            )
        try:
            pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                token=token,
            )
            if _bool("VISUAL_IMAGE_CPU_OFFLOAD", True):
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to("cuda")
        except Exception as exc:
            raise ImageGenerationError(
                "Could not load FLUX. Accept the model terms for the HF_TOKEN account: "
                f"{exc}"
            ) from exc
        self._pipeline = pipeline
        return pipeline

    def generate(self, output_dir: Path) -> tuple[KeyframeAsset, ...]:
        pipeline = self._load()
        try:
            import torch
        except ImportError as exc:
            raise ImageGenerationError("PyTorch is missing from the visual worker") from exc

        def infer(prompt: str, _negative: str, seed: int) -> Image.Image:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return pipeline(
                prompt=prompt,
                width=self.plan.width,
                height=self.plan.height,
                guidance_scale=0.0,
                num_inference_steps=self.steps,
                max_sequence_length=256,
                generator=generator,
            ).images[0]

        return _materialize_assets(
            plan=self.plan,
            output_dir=output_dir,
            backend="flux",
            model=self.model_id,
            steps=self.steps,
            infer=infer,
        )


class SDXLLightningKeyframeGenerator:
    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.base_model = os.getenv(
            "VISUAL_SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
        ).strip()
        self.repo = os.getenv("VISUAL_SDXL_LIGHTNING_REPO", "ByteDance/SDXL-Lightning").strip()
        self.checkpoint = os.getenv(
            "VISUAL_SDXL_LIGHTNING_CHECKPOINT",
            "sdxl_lightning_4step_unet.safetensors",
        ).strip()
        self.steps = int(os.getenv("VISUAL_SDXL_LIGHTNING_STEPS", "4"))
        if self.steps not in {2, 4, 8}:
            raise ImageGenerationError("VISUAL_SDXL_LIGHTNING_STEPS must be 2, 4, or 8")
        self._pipeline: Any = None

    @property
    def model_id(self) -> str:
        return f"{self.repo}:{self.checkpoint}+{self.base_model}"

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImageGenerationError("SDXL-Lightning dependencies are missing") from exc
        try:
            unet = UNet2DConditionModel.from_config(
                self.base_model,
                subfolder="unet",
            ).to("cuda", torch.float16)
            weights = hf_hub_download(self.repo, self.checkpoint)
            unet.load_state_dict(load_file(weights, device="cuda"))
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing="trailing",
            )
            pipeline.set_progress_bar_config(disable=True)
        except Exception as exc:
            raise ImageGenerationError(f"Could not load SDXL-Lightning: {exc}") from exc
        self._pipeline = pipeline
        return pipeline

    def generate(self, output_dir: Path) -> tuple[KeyframeAsset, ...]:
        pipeline = self._load()
        try:
            import torch
        except ImportError as exc:
            raise ImageGenerationError("PyTorch is missing from the visual worker") from exc

        def infer(prompt: str, negative: str, seed: int) -> Image.Image:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            return pipeline(
                prompt=prompt,
                negative_prompt=negative,
                width=self.plan.width,
                height=self.plan.height,
                num_inference_steps=self.steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

        return _materialize_assets(
            plan=self.plan,
            output_dir=output_dir,
            backend="sdxl_lightning",
            model=self.model_id,
            steps=self.steps,
            infer=infer,
        )


def selected_image_backend() -> str:
    backend = os.getenv("VISUAL_IMAGE_BACKEND", "auto").strip().lower()
    if backend == "auto":
        return "flux" if os.getenv("HF_TOKEN") else "sdxl_lightning"
    if backend not in {"flux", "sdxl_lightning"}:
        raise ImageGenerationError(f"Unsupported production image backend: {backend}")
    return backend


def generate_keyframes(plan: VisualPlan, output_dir: Path) -> tuple[KeyframeAsset, ...]:
    backend = selected_image_backend()
    if backend == "flux":
        return FluxKeyframeGenerator(plan).generate(output_dir)
    return SDXLLightningKeyframeGenerator(plan).generate(output_dir)
