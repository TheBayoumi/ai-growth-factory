from __future__ import annotations

from dataclasses import replace
from typing import Any


_INSTALLED = False


def compile_display_free_physical_prompt_v29(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = 62,
) -> Any:
    """Remove all display vocabulary from the positive prompt; CFG negatives own exclusions."""
    from .production_visual_convergence_v29 import _compile_physical_story_prompt

    compiled = _compile_physical_story_prompt(
        director_prompt,
        director_negative_prompt,
        word_budget=word_budget,
    )
    prompt = compiled.compiled_prompt.replace(
        " no visible display surfaces",
        "",
    ).replace(
        "no visible display surfaces ",
        "",
    )
    prompt = " ".join(prompt.split())
    return replace(
        compiled,
        compiled_prompt=prompt,
        word_count=len(prompt.rstrip(".").split()),
        compiler_version="visual-compiler-v29-physical-story-cfg-display-free",
    )


def install_production_visual_prompt_cleanup_v29() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler

    visual_prompt_compiler.compile_image_prompt = compile_display_free_physical_prompt_v29
    image_generator.compile_image_prompt = compile_display_free_physical_prompt_v29
    _INSTALLED = True
