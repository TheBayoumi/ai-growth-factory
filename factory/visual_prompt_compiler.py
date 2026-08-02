from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


PROMPT_COMPILER_VERSION = "visual-compiler-v5"
_IMAGE_WORD_BUDGET = 44
_IMAGE_CHARACTER_BUDGET = 320
_MOTION_WORD_BUDGET = 48

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_FACTUAL_VISUAL_PREFIX = "factual visual:"
_UI_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:user\s+interface|interface|dashboard|gui|ui)\b", re.I), "abstract light structure"),
    (re.compile(r"\b(?:computer\s+screen|screen|monitor|display)\b", re.I), "unmarked luminous plane"),
    (re.compile(r"\b(?:tablet|phone|smartphone|laptop)\b", re.I), "plain glass object"),
    (re.compile(r"\b(?:data\s+visualization|chart|graph|diagram)\b", re.I), "spatial arrangement"),
    (re.compile(r"\b(?:control\s+panel|panel|console)\b", re.I), "minimal architectural surface"),
    (re.compile(r"\b(?:label|labels|headline|headlines|caption|captions|subtitle|subtitles)\b", re.I), ""),
)
_CAMERA_MOTION_RE = re.compile(
    r"\b(?:the\s+)?camera\s+(?:slowly\s+|gently\s+)?(?:pushes|pulls|pans|tilts|zooms|"
    r"orbits|tracks|dollies|moves|drifts|sweeps)(?:\s+(?:in|out|forward|backward|left|right|"
    r"up|down|around))?\b[^.!?]*",
    re.I,
)
_REDUNDANT_DIRECTIVE_RE = re.compile(
    r"(?:vertical\s+9\s*[:x]\s*16|704\s+by\s+1280|lower\s+32\s+percent|"
    r"separate\s+animated\s+caption|no\s+text|letters|numbers|logos?|watermarks?|"
    r"trademarks?|interface\s+labels?|signatures?|borders?)",
    re.I,
)
_GENERIC_CLAUSE_TERMS = {
    "abstract", "cinematic", "clean", "digital", "glow", "glowing", "interface",
    "lighting", "minimalist", "represents", "sphere", "system", "technology",
}
_SIGNATURE_STOP = {
    "abstract", "animate", "caption", "captions", "cinematic", "coherent", "dark",
    "editorial", "empty", "frame", "geometry", "image", "interfaces", "light",
    "lighting", "lower", "motion", "portrait", "preserve", "realistic", "reserved",
    "scene", "stable", "subject", "subtle", "text", "third",
}
_ROLE_MOTION = {
    "hook": "Reveal the main subject with one clear change in light and depth.",
    "evidence": "Move evidence elements in an orderly sequence so the comparison is readable without text.",
    "mechanism": "Animate one directional process from cause to result while preserving all objects.",
    "comparison": "Shift emphasis between the two contrasted arrangements without moving the camera.",
    "implication": "Expand subtle environmental activity around the subject to suggest broader consequences.",
    "cta": "Settle the scene into a confident final state with restrained ambient motion.",
}


@dataclass(frozen=True)
class CompiledVisualPrompt:
    director_prompt: str
    compiled_prompt: str
    negative_prompt: str
    word_count: int
    word_budget: int
    compiler_version: str = PROMPT_COMPILER_VERSION


@dataclass(frozen=True)
class CompiledMotionPrompt:
    director_motion_prompt: str
    compiled_motion_prompt: str
    word_count: int
    word_budget: int
    compiler_version: str = PROMPT_COMPILER_VERSION


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" ,.;")


def _sanitize_visual_language(value: str) -> str:
    result = value
    for pattern, replacement in _UI_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    result = re.sub(r"\b(?:written|readable|legible|typographic|textual)\b", "", result, flags=re.I)
    return _clean(result)


def _sanitize_motion_language(value: str) -> str:
    return _clean(_CAMERA_MOTION_RE.sub("", _sanitize_visual_language(value)))


def _candidate_clauses(value: str) -> list[str]:
    candidates: list[str] = []
    for sentence in _SENTENCE_RE.split(value):
        clean = _clean(sentence)
        if not clean or _REDUNDANT_DIRECTIVE_RE.search(clean):
            continue
        candidates.append(clean)
    return candidates or [_clean(value)]


def _clause_score(clause: str, index: int) -> tuple[float, int]:
    lowered = clause.casefold()
    tokens = set(_TOKEN_RE.findall(lowered))
    score = float(len(tokens - _GENERIC_CLAUSE_TERMS))
    if lowered.startswith(_FACTUAL_VISUAL_PREFIX):
        score += 100.0
    if any(marker in lowered for marker in (
        "showing", "depicting", "illustrating", "visualize", "human", "world",
        "multilingual", "satisfaction", "comparison", "response", "network", "pathway",
    )):
        score += 12.0
    if "represents the ai system" in lowered or "floating abstract sphere" in lowered:
        score -= 14.0
    if any(term in lowered for term in (
        "lighting is", "color palette", "lower 32 percent", "clean, minimalist",
    )):
        score -= 8.0
    return score, -index


def _focus_clause(clause: str) -> str:
    lowered = clause.casefold()
    if lowered.startswith(_FACTUAL_VISUAL_PREFIX):
        return _clean(clause[len(_FACTUAL_VISUAL_PREFIX):])
    for marker in ("showing ", "depicting ", "illustrating ", "displaying "):
        position = lowered.find(marker)
        if position >= 0:
            focused = clause[position + len(marker):].strip(" ,.;")
            return _clean(f"Visualize {focused}")
    return clause


def _distinctive_content(value: str) -> str:
    clauses = _candidate_clauses(_sanitize_visual_language(value))
    ranked = sorted(
        enumerate(clauses),
        key=lambda item: _clause_score(item[1], item[0]),
        reverse=True,
    )
    for _index, clause in ranked:
        if clause:
            return _focus_clause(clause)
    return "factual technology subject"


def _truncate_words(value: str, maximum: int) -> str:
    words = value.split()
    if len(words) <= maximum:
        return value
    return " ".join(words[:maximum]).rstrip(" ,.;:")


def _truncate_content(value: str, *, maximum_words: int, maximum_characters: int) -> str:
    shortened = _truncate_words(value, maximum_words)
    if len(shortened) <= maximum_characters:
        return shortened
    boundary = shortened.rfind(" ", 0, maximum_characters + 1)
    if boundary < 1:
        boundary = maximum_characters
    return shortened[:boundary].rstrip(" ,.;:")


def _signature(value: str) -> frozenset[str]:
    return frozenset(
        token for token in _TOKEN_RE.findall(value.casefold()) if token not in _SIGNATURE_STOP
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def validate_compiled_prompt_diversity(scenes: Iterable[Any]) -> None:
    """Reject plans whose executable prompts collapse into one repeated visual motif."""
    image_prompts: list[tuple[int, str, frozenset[str]]] = []
    motion_prompts: list[tuple[int, str, frozenset[str]]] = []
    for scene in scenes:
        image = compile_image_prompt(scene.image_prompt, scene.negative_prompt)
        image_prompts.append(
            (scene.scene_index, image.compiled_prompt, _signature(image.compiled_prompt))
        )
        if scene.generation_mode == "wan_i2v":
            motion = compile_motion_prompt(
                scene.motion_prompt,
                semantic_context=scene.image_prompt,
                role=scene.role,
            )
            motion_prompts.append(
                (scene.scene_index, motion.compiled_motion_prompt, _signature(motion.compiled_motion_prompt))
            )

    for collection, label, threshold in (
        (image_prompts, "image", 0.76),
        (motion_prompts, "motion", 0.82),
    ):
        for offset, (left_index, left_text, left_signature) in enumerate(collection):
            for right_index, right_text, right_signature in collection[offset + 1 :]:
                if (
                    left_text.casefold() == right_text.casefold()
                    or _jaccard(left_signature, right_signature) >= threshold
                ):
                    raise ValueError(
                        f"Compiled {label} prompts for scenes {left_index} and {right_index} "
                        "are not semantically distinct"
                    )


def compile_image_prompt(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _IMAGE_WORD_BUDGET,
) -> CompiledVisualPrompt:
    """Compile rich direction into a CLIP-safe prompt that preserves the unique scene idea."""
    if word_budget < 36:
        raise ValueError("image prompt word_budget must be at least 36")
    prefix = "Text-free cinematic editorial image. No screens, signs, symbols, logos, or interfaces."
    suffix = (
        "Subject high in frame. Dark empty lower third reserved for captions. "
        "Portrait framing, stable geometry, realistic light."
    )
    fixed_words = len(prefix.split()) + len(suffix.split())
    fixed_characters = len(prefix) + len(suffix) + 3
    content = _distinctive_content(director_prompt)
    content = _truncate_content(
        content,
        maximum_words=max(7, word_budget - fixed_words),
        maximum_characters=max(70, _IMAGE_CHARACTER_BUDGET - fixed_characters),
    )
    compiled = _clean(f"{prefix} {content}. {suffix}")
    if len(compiled.split()) > word_budget or len(compiled) > _IMAGE_CHARACTER_BUDGET:
        raise ValueError("Compiled image prompt exceeded the conservative CLIP budget")

    negative_parts = (
        "text letters numbers symbols captions subtitles logos watermark signage",
        "screens monitors tablets dashboards user interface UI glyphs",
        "cluttered lower third busy foreground",
        "warped anatomy duplicate people duplicate objects malformed hands",
        "blur low detail camera shake flicker borders",
        _sanitize_visual_language(director_negative_prompt),
    )
    seen: set[str] = set()
    negative_tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9-]+", " ".join(negative_parts).lower()):
        if token in seen:
            continue
        seen.add(token)
        negative_tokens.append(token)
    negative = ", ".join(negative_tokens[:55])
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=compiled,
        negative_prompt=negative,
        word_count=len(compiled.split()),
        word_budget=word_budget,
    )


def compile_motion_prompt(
    director_motion_prompt: str,
    *,
    semantic_context: str = "",
    role: str = "",
    word_budget: int = _MOTION_WORD_BUDGET,
) -> CompiledMotionPrompt:
    """Compile scene-specific motion without reinjecting the full image prompt."""
    if word_budget < 32:
        raise ValueError("motion prompt word_budget must be at least 32")
    prefix = "Locked camera. Preserve keyframe composition, subject, lighting, and geometry."
    suffix = (
        "Subtle coherent motion only. No text, screens, new objects, cuts, zoom, shake, "
        "flicker, morphing, or anatomy changes."
    )
    semantic = _truncate_words(_distinctive_content(semantic_context), 9) if semantic_context else ""
    role_instruction = _ROLE_MOTION.get(
        role.casefold(),
        "Animate one clear internal change without moving the camera.",
    )
    raw_motion = _truncate_words(
        _distinctive_content(_sanitize_motion_language(director_motion_prompt)),
        10,
    )
    fixed = len(prefix.split()) + len(suffix.split())
    content = _truncate_words(
        _clean(f"Animate {semantic}. {role_instruction} {raw_motion}"),
        max(8, word_budget - fixed),
    )
    compiled = _clean(f"{prefix} {content}. {suffix}")
    compiled = _truncate_words(compiled, word_budget)
    return CompiledMotionPrompt(
        director_motion_prompt=director_motion_prompt,
        compiled_motion_prompt=compiled,
        word_count=len(compiled.split()),
        word_budget=word_budget,
    )
