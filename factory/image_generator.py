from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from .visual_prompt import VisualPlan
from .visual_prompt_compiler import compile_image_prompt


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
    director_prompt: str = ""
    prompt_word_count: int = 0
    prompt_word_budget: int = 0
    prompt_compiler_version: str = ""
    caption_zone_detail_before: float = 0.0
    caption_zone_detail_after: float = 0.0
    caption_zone_repaired: bool = False

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


def _detail_score(image: Image.Image) -> float:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0])


def _caption_safe_zone(image: Image.Image, *, start_ratio: float = 0.68) -> tuple[Image.Image, float, float]:
    """Feather the lower third into stable negative space for later ASS captions.

    Prompt compliance alone is probabilistic. This deterministic post-process preserves
    the upper visual while softly reducing high-frequency detail and luminance toward the
    bottom edge. It adds no text and avoids an abrupt rectangular overlay.
    """
    source = image.convert("RGB")
    width, height = source.size
    start_y = max(1, min(height - 2, round(height * start_ratio)))
    zone = source.crop((0, start_y, width, height))
    before = _detail_score(zone)
    blurred = zone.filter(ImageFilter.GaussianBlur(radius=max(9.0, width / 58.0)))

    zone_height = zone.height
    blur_mask = Image.new("L", zone.size, 0)
    dark_mask = Image.new("L", zone.size, 0)
    blur_draw = ImageDraw.Draw(blur_mask)
    dark_draw = ImageDraw.Draw(dark_mask)
    denominator = max(1, zone_height - 1)
    for row in range(zone_height):
        progress = row / denominator
        eased = progress * progress * (3.0 - 2.0 * progress)
        blur_alpha = round(225 * eased)
        dark_alpha = round(112 * eased)
        blur_draw.line((0, row, width, row), fill=blur_alpha)
        dark_draw.line((0, row, width, row), fill=dark_alpha)

    softened = Image.composite(blurred, zone, blur_mask)
    darkened = Image.composite(Image.new("RGB", zone.size, (5, 7, 12)), softened, dark_mask)
    result = source.copy()
    result.paste(darkened, (0, start_y))
    after = _detail_score(darkened)
    if before >= 4.0 and after > before * 0.94:
        raise ImageGenerationError(
            "Caption-safe lower third remained too detailed after deterministic repair: "
            f"before={before:.3f}, after={after:.3f}"
        )
    return result, before, after


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
        executable = compile_image_prompt(scene.image_prompt, scene.negative_prompt)
        try:
            image = infer(
                executable.compiled_prompt,
                executable.negative_prompt,
                scene.seed,
            ).convert("RGB")
        except Exception as exc:
            raise ImageGenerationError(
                f"{backend} failed for scene {scene.scene_index}: {exc}"
            ) from exc
        if image.size != (plan.width, plan.height):
            image = image.resize((plan.width, plan.height), Image.Resampling.LANCZOS)
        image, detail_before, detail_after = _caption_safe_zone(image)
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
                prompt=executable.compiled_prompt,
                negative_prompt=executable.negative_prompt,
                director_prompt=executable.director_prompt,
                prompt_word_count=executable.word_count,
                prompt_word_budget=executable.word_budget,
                prompt_compiler_version=executable.compiler_version,
                caption_zone_detail_before=round(detail_before, 4),
                caption_zone_detail_after=round(detail_after, 4),
                caption_zone_repaired=True,
            )
        )

    manifest = {
        "backend": backend,
        "model": model,
        "steps": steps,
        "width": plan.width,
        "height": plan.height,
        "captions_or_text_requested": False,
        "prompt_compiler_version": assets[0].prompt_compiler_version if assets else None,
        "caption_safe_zone": {
            "start_ratio": 0.68,
            "deterministic_repair": True,
            "text_added": False,
        },
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
