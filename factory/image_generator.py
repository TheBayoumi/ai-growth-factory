from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


class FluxKeyframeGenerator:
    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.model_id = os.getenv(
            "VISUAL_IMAGE_MODEL",
            "black-forest-labs/FLUX.1-schnell",
        ).strip()
        self.steps = int(os.getenv("VISUAL_IMAGE_STEPS", "4"))
        if not 1 <= self.steps <= 12:
            raise ImageGenerationError("VISUAL_IMAGE_STEPS must be between 1 and 12")
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImageGenerationError(
                "FLUX dependencies are missing. Install the visual worker requirements."
            ) from exc

        token = os.getenv("HF_TOKEN") or None
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
                "Could not load FLUX keyframe model. FLUX.1-schnell requires the "
                "Hugging Face terms to be accepted and HF_TOKEN to be available to "
                f"the visual worker: {exc}"
            ) from exc
        self._pipeline = pipeline
        return pipeline

    def generate(self, output_dir: Path) -> tuple[KeyframeAsset, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = self._load()
        try:
            import torch
        except ImportError as exc:
            raise ImageGenerationError("PyTorch is missing from the visual worker") from exc

        assets: list[KeyframeAsset] = []
        hashes: list[int] = []
        for scene in self.plan.scenes:
            generator = torch.Generator(device="cpu").manual_seed(scene.seed)
            try:
                result = pipeline(
                    prompt=scene.image_prompt,
                    width=self.plan.width,
                    height=self.plan.height,
                    guidance_scale=0.0,
                    num_inference_steps=self.steps,
                    max_sequence_length=256,
                    generator=generator,
                )
                image = result.images[0].convert("RGB")
            except Exception as exc:
                raise ImageGenerationError(
                    f"FLUX failed for scene {scene.scene_index}: {exc}"
                ) from exc
            if image.size != (self.plan.width, self.plan.height):
                image = image.resize(
                    (self.plan.width, self.plan.height),
                    Image.Resampling.LANCZOS,
                )
            path = output_dir / f"scene-{scene.scene_index:02d}-keyframe.png"
            image.save(path, format="PNG", optimize=True)
            entropy, image_hash = _validate_keyframe(
                path,
                width=self.plan.width,
                height=self.plan.height,
            )
            if any(_hamming(image_hash, previous) < 18 for previous in hashes):
                raise ImageGenerationError(
                    f"Scene {scene.scene_index} keyframe is too similar to another scene"
                )
            hashes.append(image_hash)
            assets.append(
                KeyframeAsset(
                    scene_index=scene.scene_index,
                    path=path,
                    model=self.model_id,
                    seed=scene.seed,
                    width=self.plan.width,
                    height=self.plan.height,
                    sha256=_sha256(path),
                    entropy=round(entropy, 4),
                    prompt=scene.image_prompt,
                    negative_prompt=scene.negative_prompt,
                )
            )

        manifest = {
            "backend": "flux",
            "model": self.model_id,
            "steps": self.steps,
            "width": self.plan.width,
            "height": self.plan.height,
            "assets": [asset.as_dict() for asset in assets],
        }
        (output_dir / "keyframe-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return tuple(assets)


def generate_keyframes(plan: VisualPlan, output_dir: Path) -> tuple[KeyframeAsset, ...]:
    backend = os.getenv("VISUAL_IMAGE_BACKEND", "flux").strip().lower()
    if backend != "flux":
        raise ImageGenerationError(
            f"Unsupported production image backend: {backend}. Only flux is fail-closed."
        )
    return FluxKeyframeGenerator(plan).generate(output_dir)
