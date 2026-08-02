from __future__ import annotations

from typing import Any

from .visual_prompt_compiler import (
    compile_motion_prompt,
    validate_compiled_prompt_diversity,
)


_INSTALLED = False


def install_production_visual_semantics() -> None:
    """Validate executable prompt diversity and bind scene semantics to Wan motion."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import video_generator, visual_prompt

    original_construct = visual_prompt.construct_visual_plan
    original_animate = video_generator.Wan22DiffusersAnimator.animate

    def construct(*args: Any, **kwargs: Any) -> Any:
        plan = original_construct(*args, **kwargs)
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

    visual_prompt.construct_visual_plan = construct
    video_generator.Wan22DiffusersAnimator.animate = animate
    _INSTALLED = True
