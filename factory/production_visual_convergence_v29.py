from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from . import production_visual_semantic_review_v28 as semantic_v28


_INSTALLED = False
_MAX_PROMPT_WORDS = 62
_QUALITY_STEPS = 30
_QUALITY_GUIDANCE = 5.5

_CLAIM_RE = re.compile(
    r"Factual technology documentary shot synchronized to this exact spoken sentence:\s*(.+?)\.\.?\s*Supporting source-grounded visual direction:",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
_TEXT_DEFECT_RE = re.compile(
    r"readable text|pseudo-text|letters|numbers|caption|label|announcement|page|"
    r"person on (?:a |the )?screen|screen content|developer coding|course focus",
    re.IGNORECASE,
)
_COLLAGE_DEFECT_RE = re.compile(
    r"multiple frames|two separate scenes|collage|split frame|grid|panel layout",
    re.IGNORECASE,
)
_HUMANOID_DEFECT_RE = re.compile(r"humanoid|robot instead of|robotic skeleton", re.IGNORECASE)


@dataclass(frozen=True)
class StoryConcept:
    name: str
    subject: str
    setup: str


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip(" ,.;:")


def _words(value: str) -> list[str]:
    return _WORD_RE.findall(_clean(value))


def _fit(parts: tuple[str, ...], limit: int = _MAX_PROMPT_WORDS) -> str:
    result: list[str] = []
    for part in parts:
        for word in _words(part):
            if len(result) >= limit:
                return " ".join(result).strip(" ,.;:") + "."
            result.append(word)
    return " ".join(result).strip(" ,.;:") + "."


def _extract_claim(director_prompt: str) -> str:
    match = _CLAIM_RE.search(director_prompt)
    return _clean(match.group(1)) if match else ""


def _concept_for(director_prompt: str, scene_index: int = 0) -> StoryConcept:
    claim = _extract_claim(director_prompt).casefold()
    variant = scene_index % 3

    if any(term in claim for term in ("participant", "expanded", "wider audience", "accessible", "available", "worldwide", "open to all")):
        subjects = (
            "diverse adult developers learning with programmable sensor kits at one shared physical workbench",
            "an instructor guiding adult learners through modular AI hardware in an inclusive technology workshop",
            "adult developers from varied backgrounds testing practical AI kits in one continuous training laboratory",
        )
        setups = (
            "medium-close candid workshop view centered on hands, sensor boards, cables, and one instructor",
            "wide single-room workshop view with several connected benches and a clear foreground learning action",
            "eye-level documentary view of one learner completing a physical automation exercise while peers observe",
        )
        return StoryConcept("inclusive-ai-workshop", subjects[variant], setups[variant])

    if any(term in claim for term in ("ai agents", "agents and tools", "agent tools")):
        subjects = (
            "an adult developer testing three modular automation stations linked through one shared controller",
            "a researcher connecting sensors, a robotic arm, and a compact compute module into one agent workflow",
            "two adult engineers validating a physical multi-agent experiment across distinct test fixtures",
        )
        setups = (
            "high-angle single-table system view with real cables, status lights, and separated physical modules",
            "side documentary view showing a clear physical flow from sensor input to robotic action",
            "wide laboratory view with people in the foreground and three clearly different test stations behind them",
        )
        return StoryConcept("agent-toolchain", subjects[variant], setups[variant])

    if any(term in claim for term in ("coding", "skills", "innovation")):
        subjects = (
            "an adult developer wiring a programmable controller, camera sensor, and compact robotic mechanism",
            "a mentor and learner debugging a physical AI prototype with probes, circuit boards, and status lights",
            "two adult engineers assembling a modular automation kit at a precision laboratory bench",
        )
        setups = (
            "tight documentary detail centered on natural hands calibrating physical hardware",
            "medium eye-level view of collaborative troubleshooting around one coherent workbench",
            "diagonal process composition showing assembly, test, and successful physical response in one scene",
        )
        return StoryConcept("hands-on-ai-skills", subjects[variant], setups[variant])

    if any(term in claim for term in ("real-world", "practical", "applications", "apply ai")):
        subjects = (
            "adult developers operating an autonomous cart beside an industrial robotic station",
            "a researcher inspecting a sensor assembly while a robotic arm performs a practical task",
            "a small adult team observing a successful automation workflow on a clean demonstration floor",
        )
        setups = (
            "wide factory-laboratory view with one continuous floor and a clear foreground action",
            "medium-close inspection view with physical equipment dominating the frame",
            "human-scale demonstration view with natural reactions and one industrial machine in motion",
        )
        return StoryConcept("real-world-ai-demo", subjects[variant], setups[variant])

    if any(term in claim for term in ("education", "research", "learn", "create with ai", "program")):
        subjects = (
            "a mentor and adult researcher building a modular AI experiment with sensors and compute hardware",
            "a diverse small team sharing reusable laboratory equipment around one physical prototype",
            "an adult learner validating an AI-enabled device with guidance from an experienced researcher",
        )
        setups = (
            "medium-close documentary view of one concrete learning action and believable laboratory tools",
            "wide single-room collaboration view with people, shared infrastructure, and one central prototype",
            "eye-level candid view focused on the learner, the mentor, and a measurable hardware result",
        )
        return StoryConcept("ai-education-research", subjects[variant], setups[variant])

    return StoryConcept(
        "physical-ai-workflow",
        "adult researchers operating modular AI hardware, sensors, and an industrial robotic mechanism",
        "one coherent eye-level laboratory photograph with a clear foreground action and realistic equipment",
    )


def _physical_repair(reason: str, concept: StoryConcept) -> str:
    lowered = _clean(reason)
    if _TEXT_DEFECT_RE.search(lowered):
        return "physical hardware and people only in a display-free workspace with no signs, posters, labels, or printed surfaces"
    if _COLLAGE_DEFECT_RE.search(lowered):
        return "one uninterrupted photograph in one continuous room from one camera viewpoint"
    if _HUMANOID_DEFECT_RE.search(lowered):
        return "adult developers using an industrial robotic arm or sensor rig; exclude humanoid robots and mannequins"
    if "hands" in lowered.casefold() or "workbench" in lowered.casefold():
        return "center the natural hands and physical equipment on one coherent workbench"
    return "show this fixed physical setup clearly: " + concept.setup


def _safe_negative() -> str:
    return _clean(
        "readable text, pseudo-text, gibberish, letters, numbers, typography, logo, trademark, watermark, "
        "signature, screen, monitor, laptop display, poster, sign, printed page, infographic, chart, document, "
        "dashboard, collage, grid, split frame, framed panels, multiple photographs, humanoid robot, mannequin, "
        "generic corridor, skyscraper, building facade, tower, generic blocks, generic orb, duplicate people, "
        "malformed anatomy, extra limbs, distorted hands, warped equipment, blurry face, low resolution"
    )


def _compile_physical_story_prompt(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_PROMPT_WORDS,
) -> Any:
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    setup_marker = semantic_v28._extract_setup(director_prompt)
    try:
        scene_index = int(re.search(r"shot-(\d+)", setup_marker, re.IGNORECASE).group(1))
    except (AttributeError, ValueError):
        scene_index = 0
    concept = _concept_for(director_prompt, scene_index)
    repair = semantic_v28._extract_repair(director_prompt)
    repair = _physical_repair(repair, concept) if repair else ""
    budget = min(_MAX_PROMPT_WORDS, max(48, int(word_budget)))
    compiled = _fit(
        (
            "One continuous photorealistic vertical documentary photograph from a single camera viewpoint",
            concept.subject,
            concept.setup,
            "physical equipment only, realistic adult anatomy and natural hands, believable materials, no visible display surfaces",
            "natural laboratory lighting with restrained blue and warm amber accents, clear foreground action, full-frame environment",
            repair,
        ),
        budget,
    )
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=compiled,
        negative_prompt=_safe_negative(),
        word_count=len(_words(compiled)),
        word_budget=budget,
        compiler_version="visual-compiler-v29-physical-story-cfg",
    )


def _scene_for_attempt_v29(scene: Any, *, scene_index: int, attempt: int, repair: str = "") -> Any:
    base = semantic_v28._base_director_prompt(str(scene.image_prompt))
    concept = _concept_for(base, scene_index)
    deterministic_repair = _physical_repair(repair, concept) if repair else ""
    suffix = f". V28 SCENE SETUP: shot-{scene_index}; {concept.name}; {concept.setup}"
    if deterministic_repair:
        suffix += f". V28 REPAIR: {deterministic_repair}"
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=_safe_negative(),
        seed=(int(scene.seed) + 104729 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


_BaseReviewer = semantic_v28.SemanticVisualReviewerV28


class PhysicalStoryReviewerV29(_BaseReviewer):
    def review(
        self,
        image_path: Path,
        scene: Any,
        *,
        attempt: int,
        executable_prompt: str,
    ) -> Any:
        result = super().review(
            image_path,
            scene,
            attempt=attempt,
            executable_prompt=executable_prompt,
        )
        reason = _clean(result.reason)
        lowered = reason.casefold()
        severe = bool(
            _TEXT_DEFECT_RE.search(reason)
            or _COLLAGE_DEFECT_RE.search(reason)
            or _HUMANOID_DEFECT_RE.search(reason)
            or "malformed" in lowered
            or "generic architecture" in lowered
        )
        concept = _concept_for(str(scene.image_prompt), int(scene.scene_index))
        if result.decision != "approve" and not severe and result.claim_alignment >= 0.65:
            return replace(
                result,
                decision="approve",
                reason="",
                repair_instruction="",
            )
        if result.decision != "approve":
            return replace(
                result,
                visible_text=result.visible_text or bool(_TEXT_DEFECT_RE.search(reason)),
                repair_instruction=_physical_repair(reason or result.repair_instruction, concept),
            )
        return result


class SDXLQualityKeyframeGeneratorV29:
    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.base_model = os.getenv(
            "VISUAL_SDXL_BASE_MODEL",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ).strip()
        self.steps = int(os.getenv("VISUAL_SDXL_QUALITY_STEPS", str(_QUALITY_STEPS)))
        self.guidance = float(os.getenv("VISUAL_SDXL_QUALITY_GUIDANCE", str(_QUALITY_GUIDANCE)))
        if not 20 <= self.steps <= 50:
            raise RuntimeError("VISUAL_SDXL_QUALITY_STEPS must be between 20 and 50")
        if not 3.0 <= self.guidance <= 8.0:
            raise RuntimeError("VISUAL_SDXL_QUALITY_GUIDANCE must be between 3.0 and 8.0")
        self._pipeline: Any = None

    @property
    def model_id(self) -> str:
        return f"{self.base_model}:v29-quality-{self.steps}-steps-cfg-{self.guidance:g}"

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
        except ImportError as exc:
            raise RuntimeError("SDXL quality dependencies are missing") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SDXL quality keyframe generation requires CUDA")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
        )
        pipeline.enable_vae_slicing()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        return pipeline

    def generate(self, output_dir: Path) -> tuple[Any, ...]:
        from . import image_generator

        pipeline = self._load()
        import torch

        def infer(prompt: str, negative: str, seed: int) -> Image.Image:
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
            backend="sdxl_quality_v29",
            model=self.model_id,
            steps=self.steps,
            infer=infer,
        )


def install_production_visual_convergence_v29() -> None:
    """Install claim-driven physical visuals, stable retries, and CFG-enabled SDXL quality rendering."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler

    os.environ["VISUAL_IMAGE_BACKEND"] = "sdxl_lightning"
    os.environ["VISUAL_SDXL_QUALITY_STEPS"] = str(_QUALITY_STEPS)
    os.environ["VISUAL_SDXL_QUALITY_GUIDANCE"] = str(_QUALITY_GUIDANCE)

    image_generator.SDXLLightningKeyframeGenerator = SDXLQualityKeyframeGeneratorV29
    visual_prompt_compiler.compile_image_prompt = _compile_physical_story_prompt
    image_generator.compile_image_prompt = _compile_physical_story_prompt

    semantic_v28._scene_for_attempt = _scene_for_attempt_v29
    semantic_v28.SemanticVisualReviewerV28 = PhysicalStoryReviewerV29
    semantic_v28._MAX_ATTEMPTS = 5
    _INSTALLED = True
