from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .feeds import SourceItem, source_authority
from .models import VideoPackage


_INSTALLED = False
_SCENE_HEADING_MAX_WORDS = 5
_SCENE_BODY_MAX_WORDS = 18
_SCENE_BODY_MAX_REPAIRABLE_WORDS = 36
_V65_RELEASE_ACTOR_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9&.+-]*(?:\s+[A-Z][A-Za-z0-9&.+-]*){0,4})\s+"
    r"(?:(?:has|have|just)\s+)?(?:announc(?:ed|es)|launch(?:ed|es)|releas(?:ed|es)|"
    r"introduc(?:ed|es)|unveil(?:ed|s)|publish(?:ed|es)|open-sourc(?:ed|es))\b",
    re.IGNORECASE,
)


def _finish_sentence(value: str) -> str:
    clean = " ".join(str(value or "").split()).strip(" ,;:—-")
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


def _compact_scene_text(value: object, *, maximum: int) -> str:
    words = str(value or "").split()
    if len(words) <= maximum:
        return " ".join(words)
    return " ".join(words[:maximum]).strip(" ,;:—-")


def normalize_final_raw_package_v65(
    raw: dict[str, Any],
    sources: Sequence[SourceItem],
) -> dict[str, Any]:
    """Apply the final deterministic capacity boundary after all earlier LLM wrappers.

    Only bounded deletion is allowed. Source URLs, publishers, indices, measurements, and all
    non-scene fields are preserved. Grossly overlong scene bodies remain untouched so the strict
    package validator can reject them instead of silently discarding substantial content.
    """
    from .production_package_boundary_v54 import normalize_raw_package_boundary_v54

    corrected = normalize_raw_package_boundary_v54(dict(raw), sources)
    scenes = corrected.get("scenes")
    if not isinstance(scenes, list):
        return corrected

    changed = False
    normalized: list[Any] = []
    for value in scenes:
        if not isinstance(value, dict):
            normalized.append(value)
            continue
        scene = dict(value)
        heading = " ".join(str(scene.get("heading") or "").split())
        body = " ".join(str(scene.get("body") or "").split())
        final_heading = _compact_scene_text(heading, maximum=_SCENE_HEADING_MAX_WORDS)
        final_body = body
        if _SCENE_BODY_MAX_WORDS < len(body.split()) <= _SCENE_BODY_MAX_REPAIRABLE_WORDS:
            final_body = _finish_sentence(
                _compact_scene_text(body, maximum=_SCENE_BODY_MAX_WORDS)
            )
        if final_heading != heading:
            scene["heading"] = final_heading
            changed = True
        if final_body != body:
            scene["body"] = final_body
            changed = True
        normalized.append(scene)

    if not changed:
        return corrected
    result = dict(corrected)
    result["scenes"] = normalized
    return result


def _selected_primary_source(package: VideoPackage, sources: Sequence[SourceItem]) -> SourceItem | None:
    if not package.source_urls:
        return None
    primary_url = package.source_urls[0]
    return next((source for source in sources if source.url == primary_url), None)


def _source_title_release_actor(source: SourceItem) -> str:
    """Return a release actor only when the supplied source title itself states the action."""
    match = _V65_RELEASE_ACTOR_RE.search(source.title)
    return " ".join(match.group(1).split()) if match else ""


def _replace_host_actor_with_title_actor(text: str, source: SourceItem) -> str:
    """Correct only the known host-as-actor error using actor text already in the source title."""
    from .production_content import _authority_tokens

    title_actor = _source_title_release_actor(source)
    if not title_actor:
        return text
    match = _V65_RELEASE_ACTOR_RE.search(text)
    if not match:
        return text

    actor = match.group(1)
    actor_tokens = _authority_tokens(actor)
    publisher_tokens = _authority_tokens(source.publisher)
    title_actor_tokens = _authority_tokens(title_actor)
    if not actor_tokens or not publisher_tokens or not title_actor_tokens:
        return text
    if not (actor_tokens & publisher_tokens):
        return text

    # The source title is the evidence for the replacement. Never derive an actor from a model
    # guess, URL, or article author when the title does not explicitly state the release action.
    start, end = match.span(1)
    return text[:start] + title_actor + text[end:]


def repair_hosting_publisher_attribution_v65(
    package: VideoPackage,
    sources: Sequence[SourceItem],
) -> VideoPackage:
    primary = _selected_primary_source(package, sources)
    if primary is None:
        return package
    authority = source_authority(primary)
    if not authority or authority.casefold() == primary.publisher.casefold():
        return package
    if not _source_title_release_actor(primary):
        return package

    title = _replace_host_actor_with_title_actor(package.title, primary)
    narration = _replace_host_actor_with_title_actor(package.narration, primary)
    if title == package.title and narration == package.narration:
        return package
    return replace(package, title=title, narration=narration)


def stabilize_final_package_v65(
    package: VideoPackage,
    sources: Sequence[SourceItem],
) -> VideoPackage:
    """Converge bounded scene capacity and source-title-grounded actor attribution only."""
    scenes = []
    for scene in package.scenes:
        body = " ".join(scene.body.split())
        if _SCENE_BODY_MAX_WORDS < len(body.split()) <= _SCENE_BODY_MAX_REPAIRABLE_WORDS:
            body = _finish_sentence(_compact_scene_text(body, maximum=_SCENE_BODY_MAX_WORDS))
        heading = _compact_scene_text(scene.heading, maximum=_SCENE_HEADING_MAX_WORDS)
        scenes.append(replace(scene, heading=heading, body=body))
    stabilized = replace(package, scenes=scenes)
    return repair_hosting_publisher_attribution_v65(stabilized, sources)


def install_production_editorial_boundary_v65() -> None:
    """Install the final package/editorial boundary after legacy production wrappers."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm, production_content

    current_package_from_raw = local_llm._package_from_raw
    current_ground = production_content._ground_generic_copy

    if not getattr(current_package_from_raw, "_agf_v65", False):
        def package_from_raw_v65(
            settings: Any,
            sources: list[SourceItem],
            raw: dict[str, Any],
        ) -> Any:
            return current_package_from_raw(
                settings,
                sources,
                normalize_final_raw_package_v65(raw, sources),
            )

        package_from_raw_v65._agf_v65 = True  # type: ignore[attr-defined]
        local_llm._package_from_raw = package_from_raw_v65

    if not getattr(current_ground, "_agf_v65", False):
        def ground_v65(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
            grounded = current_ground(package, sources)
            return stabilize_final_package_v65(grounded, sources)

        ground_v65._agf_v65 = True  # type: ignore[attr-defined]
        production_content._ground_generic_copy = ground_v65

    _INSTALLED = True
