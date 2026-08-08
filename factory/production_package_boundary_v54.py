from __future__ import annotations

from typing import Any, Sequence

from .feeds import SourceItem


_INSTALLED = False
_SCENE_HEADING_MAX_WORDS = 5
_SCENE_BODY_MAX_WORDS = 18
_SCENE_BODY_MAX_REPAIRABLE_WORDS = 36


def _finish_sentence(value: str) -> str:
    clean = " ".join(value.split()).strip(" ,;:—-")
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


def _compact_words(value: object, *, maximum: int) -> str:
    words = str(value or "").split()
    if len(words) <= maximum:
        return " ".join(words)
    return " ".join(words[:maximum]).strip(" ,;:—-")


def normalize_raw_package_boundary_v54(
    raw: dict[str, Any],
    sources: Sequence[SourceItem],
) -> dict[str, Any]:
    """Normalize harmless JSON capacity overshoots before strict package validation.

    The generation model occasionally returns a 19-22 word scene body despite an 18-word
    contract. Rejecting an otherwise coherent current story and trying another trend is expensive
    release-engineering noise. This adapter performs only deterministic deletion; it never adds a
    claim, changes source indices, substitutes evidence, or relaxes the downstream validator.
    """
    from .production_package_capacity_v46 import stabilize_raw_package_capacity

    corrected = stabilize_raw_package_capacity(dict(raw), list(sources))
    scenes = corrected.get("scenes")
    if not isinstance(scenes, list):
        return corrected

    normalized_scenes: list[Any] = []
    changed = False
    for value in scenes:
        if not isinstance(value, dict):
            normalized_scenes.append(value)
            continue
        scene = dict(value)
        heading = " ".join(str(scene.get("heading") or "").split())
        compact_heading = _compact_words(heading, maximum=_SCENE_HEADING_MAX_WORDS)
        body = " ".join(str(scene.get("body") or "").split())
        compact_body = body
        if _SCENE_BODY_MAX_WORDS < len(body.split()) <= _SCENE_BODY_MAX_REPAIRABLE_WORDS:
            compact_body = _finish_sentence(
                _compact_words(body, maximum=_SCENE_BODY_MAX_WORDS)
            )
        if compact_heading != heading:
            scene["heading"] = compact_heading
            changed = True
        if compact_body != body:
            scene["body"] = compact_body
            changed = True
        normalized_scenes.append(scene)

    if not changed:
        return corrected
    result = dict(corrected)
    result["scenes"] = normalized_scenes
    return result


def install_production_package_boundary_v54() -> None:
    """Install the final raw JSON capacity boundary outside all earlier LLM adapters."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    current = local_llm._package_from_raw
    if getattr(current, "_agf_v54", False):
        _INSTALLED = True
        return

    def package_from_raw_v54(
        settings: Any,
        sources: list[SourceItem],
        raw: dict[str, Any],
    ) -> Any:
        return current(
            settings,
            sources,
            normalize_raw_package_boundary_v54(raw, sources),
        )

    package_from_raw_v54._agf_v54 = True  # type: ignore[attr-defined]
    local_llm._package_from_raw = package_from_raw_v54
    _INSTALLED = True
