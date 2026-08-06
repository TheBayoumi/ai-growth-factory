from __future__ import annotations

from dataclasses import replace
from typing import Any

from .visual_prompt_compiler import (
    compile_image_prompt,
    compile_motion_prompt,
    validate_compiled_prompt_diversity,
)


_INSTALLED = False
_DIVERSITY_RULES = """

PRODUCTION VISUAL DIVERSITY RULES:
- Continuity comes from palette, material language, lighting, and one small secondary accent. It must not come from repeating the same primary object.
- Every scene needs a different primary subject, environment, composition, silhouette, and explanatory grammar.
- Never use a glowing sphere, orb, floating interface, dashboard, or generic AI core as the primary subject in more than one scene.
- Hook: show a concrete human-scale moment or consequence.
- Evidence: show a measurable contrast through physical arrangement, quantity, distance, or grouping without text or numbers.
- Mechanism: show a directional process, pathway, handoff, or cause-to-result sequence.
- Comparison: show two visibly different arrangements in the same frame.
- Implication: widen the environment or scale while keeping the factual subject concrete.
- CTA: end with a human-scale outcome or unresolved practical choice, not a generic logo-like object.
- Motion prompts must be unique to the scene and animate its specific subject. Do not repeat one generic rotation or glow instruction.
- The recurring continuity anchor must remain secondary and occupy less than 15 percent of the composition.
""".strip()

_TREATMENTS = {
    "hook": (
        "Create an immediate human-scale opening moment with one unmistakable consequence in the foreground. Use an asymmetric close composition, strong depth separation, and one concrete subject rather than a symbolic AI object.",
        "Reveal the concrete consequence through one controlled change in depth, illumination, or subject action while every object keeps its shape.",
    ),
    "evidence": (
        "Build a physical evidence composition using clearly different groups, quantities, distances, or material states. The contrast must be visible without written labels, numbers, screens, or infographic panels.",
        "Progressively emphasize the evidence groups in a readable sequence, preserving their count, position, and identity throughout the shot.",
    ),
    "mechanism": (
        "Show a directional cause-to-result mechanism as a tangible pathway, handoff, laboratory process, or sequence of connected physical stages. The primary subject must be the process itself, not a generic glowing core.",
        "Animate the mechanism from input through intermediate stages to outcome, with one directional flow and no new objects appearing.",
    ),
    "comparison": (
        "Construct a split or mirrored composition with two visibly different workflows, environments, or outcomes. Each side must have a distinct silhouette and spatial organization while sharing only the secondary palette and lighting language.",
        "Shift visual emphasis from the first arrangement to the second, keeping both comparison sides stable and continuously visible.",
    ),
    "implication": (
        "Use a wider environmental composition that shows how the factual change affects people, infrastructure, research practice, or scale. Keep the consequence concrete and distribute activity across foreground, middle distance, and background.",
        "Expand subtle activity outward from the main subject into the wider environment without changing the camera or inventing additional claims.",
    ),
    "cta": (
        "End on a human-scale practical decision or unresolved next step. Use a calm final composition with one person, tool, pathway, or open workspace that visually invites evaluation rather than resembling a logo or generic technology emblem.",
        "Settle the scene into a deliberate final state, leaving one practical choice visually unresolved and preserving all subject details.",
    ),
}
_INDEX_FALLBACKS = (
    "foreground consequence",
    "grouped physical evidence",
    "directional mechanism",
    "side-by-side comparison",
    "wide environmental implication",
    "human-scale decision",
)


def _validate_deterministic_repair(scenes: tuple[Any, ...]) -> None:
    """Validate the generated executable prompts without shared boilerplate bias.

    The normal language-model plan uses semantic Jaccard validation. A deterministic
    fallback already guarantees a different composition grammar per index; at this
    boundary we verify that those grammars survived compilation as genuinely different
    executable image and motion strings. Shared safety prefixes and caption-safe suffixes
    must not be counted as semantic duplication.
    """
    image_prompts = [
        compile_image_prompt(scene.image_prompt, scene.negative_prompt).compiled_prompt.casefold()
        for scene in scenes
    ]
    motion_prompts = [
        compile_motion_prompt(
            scene.motion_prompt,
            semantic_context=scene.image_prompt,
            role=scene.role,
        ).compiled_motion_prompt.casefold()
        for scene in scenes
        if scene.generation_mode == "wan_i2v"
    ]
    if len(set(image_prompts)) != len(image_prompts):
        raise ValueError("Deterministic scene repair produced duplicate executable image prompts")
    if len(set(motion_prompts)) != len(motion_prompts):
        raise ValueError("Deterministic scene repair produced duplicate executable motion prompts")
    forbidden = ("floating abstract sphere", "generic ai core")
    if any(term in prompt for prompt in image_prompts for term in forbidden):
        raise ValueError("Deterministic scene repair retained a forbidden generic primary motif")


def _scene_specific_repair(plan: Any, package: Any) -> Any:
    """Convert validated scene facts into deterministic, role-specific visual grammar.

    The language model remains responsible for topic selection, factual package content,
    scene roles, global style, palette, lighting, and continuity. This fallback is invoked
    only when its executable prompts still collapse. It preserves each package scene's
    heading/body/visual while guaranteeing a different composition and motion grammar for
    every scene before GPU generation.
    """
    package_scenes = list(package.scenes)
    repaired = []
    for index, scene in enumerate(plan.scenes):
        source_scene = package_scenes[index]
        role = str(scene.role).strip().lower()
        image_treatment, motion_treatment = _TREATMENTS.get(
            role,
            _TREATMENTS["implication"],
        )
        unique_grammar = _INDEX_FALLBACKS[index % len(_INDEX_FALLBACKS)]
        image_prompt = (
            f"Factual visual: {source_scene.heading}; {source_scene.visual}; "
            f"{source_scene.body}; composition grammar {unique_grammar}. "
            f"{image_treatment} "
            "Keep all generated media free of written text, labels, logos, and watermarks. "
            "Place the primary subject in the upper two-thirds and keep the lower third visually quiet for separate captions."
        )
        motion_prompt = (
            f"Animate the specific subject of {source_scene.heading}. {motion_treatment} "
            f"Preserve the {unique_grammar} composition, subject identity, lighting, and geometry."
        )
        repaired.append(
            replace(
                scene,
                image_prompt=image_prompt,
                motion_prompt=motion_prompt,
                continuity_anchor=(
                    f"secondary palette accent for {source_scene.heading}; never the primary subject"
                ),
            )
        )
    repaired_plan = replace(plan, scenes=tuple(repaired))
    _validate_deterministic_repair(repaired_plan.scenes)
    return repaired_plan


def install_production_visual_semantics() -> None:
    """Validate executable prompt diversity and bind scene semantics to Wan motion."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import video_generator, visual_prompt

    original_director_prompt = visual_prompt._director_prompt
    original_validate = visual_prompt._validate_and_normalize
    original_animate = video_generator.Wan22DiffusersAnimator.animate

    def director_prompt(payload: dict[str, Any], *, feedback: str = "") -> str:
        return original_director_prompt(payload, feedback=feedback) + "\n\n" + _DIVERSITY_RULES

    def validate(raw: dict[str, Any], **kwargs: Any) -> Any:
        plan = original_validate(raw, **kwargs)
        try:
            validate_compiled_prompt_diversity(plan.scenes)
            return plan
        except ValueError:
            package = kwargs.get("package")
            if package is None:
                raise
            return _scene_specific_repair(plan, package)

    def animate(self: Any, scene: Any, keyframe: Any, output: Any) -> Any:
        original_compiler = video_generator.compile_motion_prompt

        def scene_compiler(director_motion_prompt: str, **_kwargs: Any) -> Any:
            return compile_motion_prompt(
                director_motion_prompt,
                # The reviewed keyframe prompt contains the literal subject/action contract;
                # the director prompt starts with generic boilerplate and previously displaced it.
                semantic_context=keyframe.prompt,
                role=scene.role,
            )

        video_generator.compile_motion_prompt = scene_compiler
        try:
            return original_animate(self, scene, keyframe, output)
        finally:
            video_generator.compile_motion_prompt = original_compiler

    visual_prompt._director_prompt = director_prompt
    visual_prompt._validate_and_normalize = validate
    video_generator.Wan22DiffusersAnimator.animate = animate
    _INSTALLED = True
