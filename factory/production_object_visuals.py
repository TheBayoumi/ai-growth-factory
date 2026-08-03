from __future__ import annotations

import re
from typing import Any

from .visual_prompt_compiler import CompiledVisualPrompt


_INSTALLED = False
_COMPILER_VERSION = "visual-compiler-v8-coherent-clip-budget"
_SPACE_RE = re.compile(r"\s+")
_FORBIDDEN_POSITIVE_RE = re.compile(
    r"\b(?:no|not|without|text|letter|number|symbol|caption|subtitle|logo|watermark|"
    r"person|people|human|researcher|face|portrait|body|hand|phone|device|screen|"
    r"monitor|tablet|laptop|poster|book|document|sign|frame|panel|grid|collage|"
    r"interface|dashboard|computer)\b",
    re.IGNORECASE,
)

_ROLE_SUBJECTS = {
    "hook": "A modular bridge joins separated luminous blocks.",
    "evidence": "Scattered geometric modules converge into one orderly assembly.",
    "mechanism": "Precision components channel light through a single rising pathway.",
    "comparison": "One modular column rises from rough dark blocks into aligned glowing blocks.",
    "implication": "Connected modules expand across a clean architectural field.",
    "cta": "Distinct geometric pieces interlock around an open central platform.",
}

_NEGATIVE_PROMPT = (
    "text, letters, numbers, words, typography, pseudo-text, gibberish, logo, watermark, "
    "signature, signage, people, person, face, portrait, body, hands, phone, laptop, "
    "computer, screen, monitor, tablet, book, paper, document, poster, frame, panel, grid, "
    "collage, interface, dashboard, UI, duplicated objects, malformed geometry, cluttered "
    "lower third, busy foreground, blur, low resolution, split scene, diptych, multiple rooms"
)


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" ,.;")


def _role(director_prompt: str) -> str:
    lowered = director_prompt.casefold()
    markers = (
        ("foreground consequence", "hook"),
        ("grouped physical evidence", "evidence"),
        ("directional mechanism", "mechanism"),
        ("side-by-side comparison", "comparison"),
        ("wide environmental implication", "implication"),
        ("human-scale decision", "cta"),
    )
    for marker, role in markers:
        if marker in lowered:
            return role
    if any(term in lowered for term in ("performance", "process", "pathway", "mechanism")):
        return "mechanism"
    if any(term in lowered for term in ("compare", "contrast", "wide tasks", "range of tasks")):
        return "comparison"
    if any(term in lowered for term in ("collaboration", "open source", "open-source")):
        return "cta"
    if any(term in lowered for term in ("scale", "broader", "infrastructure")):
        return "implication"
    return "evidence"


def compile_object_only_image_prompt(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = 44,
) -> CompiledVisualPrompt:
    """Compile one short positive-only physical metaphor for SDXL-Lightning.

    SDXL-Lightning's official inference contract uses classifier-free guidance set to zero,
    so a negative prompt cannot be relied upon to suppress unwanted subjects. The positive
    prompt therefore contains only desired visible objects and composition. All prohibited
    concepts remain in the audit negative prompt, while the executable positive prompt stays
    well below CLIP's 77-token ceiling. The comparison role uses one continuous object rather
    than two opposing arrangements so the generated image remains one coherent scene.
    """
    del word_budget
    role = _role(director_prompt)
    subject = _ROLE_SUBJECTS[role]
    compiled = _clean(
        "Vertical 3D scene, unmarked matte forms, blue and amber light. "
        f"{subject} Subject above center; empty dark lower third; soft volumetric light."
    )
    words = compiled.split()
    if len(words) > 36 or len(compiled) > 240:
        raise ValueError("Object-only production prompt exceeded its conservative CLIP budget")
    forbidden = _FORBIDDEN_POSITIVE_RE.search(compiled)
    if forbidden:
        raise ValueError(
            "Object-only production prompt contains prohibited positive vocabulary: "
            + forbidden.group(0)
        )
    negative = _clean(f"{_NEGATIVE_PROMPT}, {director_negative_prompt}")
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=compiled,
        negative_prompt=negative,
        word_count=len(words),
        word_budget=36,
        compiler_version=_COMPILER_VERSION,
    )


def install_production_object_visuals() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator, visual_prompt_compiler

    visual_prompt_compiler.compile_image_prompt = compile_object_only_image_prompt
    image_generator.compile_image_prompt = compile_object_only_image_prompt
    _INSTALLED = True
