from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from . import production_visual_semantic_review_v28 as semantic_v28
from .production_visual_convergence_v29 import SDXLQualityKeyframeGeneratorV29
from .visual_storyboard_v30 import clean, storyboard_for


_INSTALLED = False
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
_REPAIR_RE = re.compile(r"V31 REPAIR:\s*(.+)$", re.IGNORECASE)
_STORYBOARD_TAIL_RE = re.compile(r"\s*(?:\.\s*)?V30 STORYBOARD:.*$", re.IGNORECASE)
_MAX_WORDS = 52


def _words(value: str) -> list[str]:
    return _WORD_RE.findall(clean(value))


def _fit(parts: tuple[str, ...], limit: int = _MAX_WORDS) -> str:
    result: list[str] = []
    for part in parts:
        for word in _words(part):
            if len(result) >= limit:
                return " ".join(result).strip(" ,.;:") + "."
            result.append(word)
    return " ".join(result).strip(" ,.;:") + "."


def _extract_repair(value: str) -> str:
    match = _REPAIR_RE.search(value)
    return clean(match.group(1)) if match else ""


def _compact_negative() -> str:
    # Keep the complete negative contract inside SDXL's CLIP window. Long negative prompts were
    # silently truncated, dropping late constraints such as empty-room and architecture-only bans.
    return clean(
        "readable text, pseudo-text, gibberish, logo, watermark, screen, monitor, poster, chart, "
        "infographic, collage, split frame, empty room, vacant server aisle, architecture-only scene, "
        "unoccupied workspace, absent people, tiny distant people, humanoid robot, duplicate people, "
        "malformed anatomy, extra limbs, distorted hands, warped equipment, blurry face, generic corridor, "
        "generic blocks, generic orb"
    )


def _physical_repair(reason: str) -> str:
    lowered = clean(reason).casefold()
    if any(term in lowered for term in ("readable text", "pseudo-text", "letters", "numbers", "label", "screen")):
        return "Use blank unmarked equipment while every named adult and physical action remains clearly visible"
    if any(term in lowered for term in ("collage", "split", "grid", "multiple frames", "panel")):
        return "Use one uninterrupted photograph in one continuous environment from one camera"
    if any(term in lowered for term in ("malformed", "distorted", "extra limb", "broken equipment")):
        return "Show fewer foreground elements with realistic adult anatomy natural hands and mechanically plausible equipment"
    if any(
        term in lowered
        for term in (
            "no people",
            "no visible people",
            "lacks the specified people",
            "lacks the specified",
            "lacks the technical staff",
            "lacks the required",
            "subjects and actions",
            "elements or actions",
            "empty architecture",
            "server room",
            "data-storage hardware room",
        )
    ):
        return "Every named adult must fill the foreground with natural hands performing the specified physical action"
    return "Make every named adult object and physical action large clear and unmistakable in the foreground"


def compile_subject_first_prompt_v31(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_WORDS,
) -> Any:
    """Compile a CLIP-safe prompt with subject and action before environment or style."""
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    frame = storyboard_for(director_prompt)
    repair = _extract_repair(director_prompt)
    budget = min(_MAX_WORDS, max(44, int(word_budget)))
    prompt = _fit(
        (
            "Photorealistic vertical documentary photograph",
            frame.subject,
            frame.action,
            repair,
            frame.environment,
            frame.camera,
            "people and active equipment dominate the foreground",
            frame.palette,
            "one coherent full-frame scene with realistic anatomy and natural hands",
        ),
        budget,
    )
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=prompt,
        negative_prompt=_compact_negative(),
        word_count=len(_words(prompt)),
        word_budget=budget,
        compiler_version="visual-compiler-v31-subject-first-clip-safe",
    )


def scene_for_attempt_v31(scene: Any, *, scene_index: int, attempt: int, repair: str = "") -> Any:
    # Always restart from the immutable base scene. Retry instructions replace one another and are
    # converted to a deterministic physical correction that the compiler actually consumes.
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    base = clean(_STORYBOARD_TAIL_RE.sub("", base))
    frame = storyboard_for(base, scene_index)
    suffix = f". V30 STORYBOARD: shot-{scene_index}; {frame.identity}"
    if repair:
        suffix += f". V31 REPAIR: {_physical_repair(repair)}"
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=_compact_negative(),
        seed=(int(scene.seed) + 161803 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def _token_count(tokenizer: Any, value: str) -> int:
    encoded = tokenizer(
        value,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def validate_clip_windows(pipeline: Any, prompt: str, negative_prompt: str) -> None:
    """Fail before inference rather than allowing either SDXL encoder to truncate silently."""
    from .image_generator import ImageGenerationError

    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipeline, name, None)
        if tokenizer is None:
            continue
        configured = int(getattr(tokenizer, "model_max_length", 77) or 77)
        maximum = configured if 1 <= configured <= 256 else 77
        positive_count = _token_count(tokenizer, prompt)
        negative_count = _token_count(tokenizer, negative_prompt)
        if positive_count > maximum or negative_count > maximum:
            raise ImageGenerationError(
                "SDXL CLIP window overflow: "
                f"{name} positive={positive_count}/{maximum}, negative={negative_count}/{maximum}"
            )


class CLIPSafeSDXLQualityKeyframeGeneratorV31(SDXLQualityKeyframeGeneratorV29):
    @property
    def model_id(self) -> str:
        return super().model_id + ":v31-subject-first-clip-safe"

    def generate(self, output_dir: Path) -> tuple[Any, ...]:
        from . import image_generator

        pipeline = self._load()
        import torch

        def infer(prompt: str, negative: str, seed: int) -> Image.Image:
            validate_clip_windows(pipeline, prompt, negative)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            return pipeline(
                prompt=prompt,
                negative_prompt=negative,
                width=self.plan.width,
                height=self.plan.height,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                guidance_rescale=0.5,
                generator=generator,
            ).images[0]

        return image_generator._materialize_assets(
            plan=self.plan,
            output_dir=output_dir,
            backend="sdxl_quality_v31",
            model=self.model_id,
            steps=self.steps,
            infer=infer,
        )


def install_production_visual_subject_authority_v31() -> None:
    """Install subject-first prompts, effective physical retries, and hard CLIP-window checks."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler

    image_generator.SDXLLightningKeyframeGenerator = CLIPSafeSDXLQualityKeyframeGeneratorV31
    visual_prompt_compiler.compile_image_prompt = compile_subject_first_prompt_v31
    image_generator.compile_image_prompt = compile_subject_first_prompt_v31
    semantic_v28._scene_for_attempt = scene_for_attempt_v31
    semantic_v28._MAX_ATTEMPTS = int(os.getenv("V31_VISUAL_REVIEW_ATTEMPTS", "4"))
    _INSTALLED = True
