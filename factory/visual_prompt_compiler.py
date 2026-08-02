from __future__ import annotations

import re
from dataclasses import dataclass


PROMPT_COMPILER_VERSION = "visual-compiler-v2"
_IMAGE_WORD_BUDGET = 64
_MOTION_WORD_BUDGET = 58

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_UI_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:user\s+interface|interface|dashboard|gui|ui)\b", re.I), "abstract light sculpture"),
    (re.compile(r"\b(?:computer\s+screen|screen|monitor|display)\b", re.I), "soft luminous plane"),
    (re.compile(r"\b(?:tablet|phone|smartphone|laptop)\b", re.I), "plain unmarked glass object"),
    (re.compile(r"\b(?:data\s+visualization|chart|graph|diagram)\b", re.I), "flowing abstract light"),
    (re.compile(r"\b(?:control\s+panel|panel|console)\b", re.I), "minimal architectural surface"),
    (re.compile(r"\b(?:label|labels|headline|headlines|caption|captions|subtitle|subtitles)\b", re.I), ""),
)
_REDUNDANT_DIRECTIVE_RE = re.compile(
    r"(?:vertical\s+9\s*[:x]\s*16|704\s+by\s+1280|lower\s+32\s+percent|"
    r"separate\s+animated\s+caption|no\s+text|letters|numbers|logos?|watermarks?|"
    r"trademarks?|interface\s+labels?|signatures?|borders?)",
    re.I,
)


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


def _core_sentences(value: str) -> str:
    candidates: list[str] = []
    for sentence in _SENTENCE_RE.split(value):
        clean = _clean(sentence)
        if not clean or _REDUNDANT_DIRECTIVE_RE.search(clean):
            continue
        candidates.append(clean)
    return ". ".join(candidates) or _clean(value)


def _truncate_words(value: str, maximum: int) -> str:
    words = value.split()
    if len(words) <= maximum:
        return value
    return " ".join(words[:maximum]).rstrip(" ,.;:")


def compile_image_prompt(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _IMAGE_WORD_BUDGET,
) -> CompiledVisualPrompt:
    """Compile a rich director prompt into a CLIP-safe, text-resistant image prompt.

    The director output remains untouched in the audit artifact. The executable prompt
    places the critical composition and text-free constraints before scene content and
    fits below SDXL's practical CLIP token ceiling.
    """
    if word_budget < 40:
        raise ValueError("image prompt word_budget must be at least 40")
    prefix = (
        "Text-free cinematic editorial image, without screens, signage, symbols, logos, "
        "or interface elements."
    )
    suffix = (
        "Subject framed in the upper two-thirds. Lower third is simple dark empty negative "
        "space for later captions. Vertical 9:16, stable geometry, realistic lighting."
    )
    reserved = len(prefix.split()) + len(suffix.split())
    content_budget = max(8, word_budget - reserved)
    content = _sanitize_visual_language(_core_sentences(director_prompt))
    content = _truncate_words(content, content_budget)
    compiled = _clean(f"{prefix} {content}. {suffix}")
    compiled = _truncate_words(compiled, word_budget)

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
    word_budget: int = _MOTION_WORD_BUDGET,
) -> CompiledMotionPrompt:
    """Compile motion instructions without re-injecting the long image prompt."""
    if word_budget < 35:
        raise ValueError("motion prompt word_budget must be at least 35")
    prefix = (
        "Locked camera. Preserve the exact keyframe composition, subject identity, lighting, "
        "and geometry."
    )
    suffix = (
        "Use subtle coherent environmental motion only. No text, screens, new objects, cuts, "
        "zoom, shake, flicker, morphing, or anatomy changes."
    )
    reserved = len(prefix.split()) + len(suffix.split())
    content = _sanitize_visual_language(_core_sentences(director_motion_prompt))
    content = _truncate_words(content, max(6, word_budget - reserved))
    compiled = _clean(f"{prefix} {content}. {suffix}")
    compiled = _truncate_words(compiled, word_budget)
    return CompiledMotionPrompt(
        director_motion_prompt=director_motion_prompt,
        compiled_motion_prompt=compiled,
        word_count=len(compiled.split()),
        word_budget=word_budget,
    )
