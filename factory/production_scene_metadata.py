from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_INSTALLED = False
_HEADING_MAX_WORDS = 5
_HEADING_MAX_CHARS = 60
_BODY_MAX_WORDS = 18
_BODY_MAX_CHARS = 240
_VISUAL_MAX_CHARS = 400


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _bounded_field(
    value: Any,
    *,
    scene_index: int,
    field: str,
    max_chars: int,
    max_words: int | None = None,
) -> str:
    from .local_llm import LocalLLMError

    text = _normalized_text(value)
    if not text:
        raise LocalLLMError(f"Scene {scene_index} {field} must not be empty")

    word_count = len(text.split())
    if max_words is not None and word_count > max_words:
        raise LocalLLMError(
            f"Scene {scene_index} {field} exceeds {max_words}-word limit: {word_count}"
        )
    if len(text) > max_chars:
        raise LocalLLMError(
            f"Scene {scene_index} {field} exceeds {max_chars}-character limit: {len(text)}"
        )
    return text


def enforce_production_scene_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate scene copy before the legacy parser can silently slice it.

    The base package parser historically clipped body text at character 180. That turned
    a valid phrase such as ``broader community`` into ``bro`` and allowed corrupted audit
    metadata to pass every downstream gate. Production now rejects overlong scene fields so
    the existing bounded package-repair loop rewrites them as complete, concise statements.
    """
    from .local_llm import LocalLLMError

    scenes = raw.get("scenes")
    if scenes is None:
        return raw
    if not isinstance(scenes, list):
        raise LocalLLMError("scenes must be an array of six objects")

    corrected_scenes: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            raise LocalLLMError(f"Scene {scene_index} must be an object")
        corrected_scene = dict(scene)
        corrected_scene["heading"] = _bounded_field(
            scene.get("heading"),
            scene_index=scene_index,
            field="heading",
            max_words=_HEADING_MAX_WORDS,
            max_chars=_HEADING_MAX_CHARS,
        )
        corrected_scene["body"] = _bounded_field(
            scene.get("body"),
            scene_index=scene_index,
            field="body",
            max_words=_BODY_MAX_WORDS,
            max_chars=_BODY_MAX_CHARS,
        )
        corrected_scene["visual"] = _bounded_field(
            scene.get("visual"),
            scene_index=scene_index,
            field="visual",
            max_chars=_VISUAL_MAX_CHARS,
        )
        corrected_scenes.append(corrected_scene)

    corrected = dict(raw)
    corrected["scenes"] = corrected_scenes
    return corrected


def install_production_scene_metadata() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original_package_from_raw = local_llm._package_from_raw

    def package_from_raw(settings: Any, sources: Any, raw: dict[str, Any]) -> Any:
        return original_package_from_raw(
            settings,
            sources,
            enforce_production_scene_metadata(raw),
        )

    local_llm._package_from_raw = package_from_raw
    _INSTALLED = True
