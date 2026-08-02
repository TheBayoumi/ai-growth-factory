from __future__ import annotations

from typing import Any

from .visual_prompt_compiler import (
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
        validate_compiled_prompt_diversity(plan.scenes)
        return plan

    def animate(self: Any, scene: Any, keyframe: Any, output: Any) -> Any:
        original_compiler = video_generator.compile_motion_prompt

        def scene_compiler(director_motion_prompt: str, **_kwargs: Any) -> Any:
            return compile_motion_prompt(
                director_motion_prompt,
                semantic_context=scene.image_prompt,
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
