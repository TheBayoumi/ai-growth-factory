from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from . import local_llm
from .config import Settings
from .feeds import SourceItem
from .models import VideoPackage
from .policy import Strategy


_GENERATION_LOCK = threading.RLock()


class _UnsupportedSceneAttribution(local_llm.LocalLLMError):
    """Raised when no selected source can support one or more scene claims."""


def _selected_source_records(
    source_urls: list[str],
    sources: list[SourceItem],
) -> list[dict[str, str]]:
    by_url = {source.url: source for source in sources}
    records: list[dict[str, str]] = []
    for url in source_urls:
        source = by_url.get(url)
        if source is None:
            raise local_llm.LocalLLMError(
                f"Selected source URL is absent from the supplied evidence set: {url}"
            )
        records.append(
            {
                "source_url": url,
                "publisher": source.publisher,
                "title": source.title[:300],
                "summary": source.summary[:1200],
            }
        )
    return records


def _scene_records(scenes_raw: list[Any]) -> list[dict[str, Any]]:
    if len(scenes_raw) != 6:
        raise local_llm.LocalLLMError("Exactly six scenes are required for attribution")
    records: list[dict[str, Any]] = []
    for scene_id, scene in enumerate(scenes_raw):
        if not isinstance(scene, dict):
            raise local_llm.LocalLLMError(
                f"Scene {scene_id} must be a JSON object for attribution"
            )
        records.append(
            {
                "scene_id": scene_id,
                "heading": str(scene.get("heading") or "")[:120],
                "body": str(scene.get("body") or "")[:300],
                "visual": str(scene.get("visual") or "")[:500],
            }
        )
    return records


def _attribution_schema(source_urls: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_id": {"type": "integer", "minimum": 0, "maximum": 5},
                        "source_url": {"type": "string", "enum": source_urls},
                    },
                    "required": ["scene_id", "source_url"],
                    "additionalProperties": False,
                },
            },
            "unsupported_scene_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 5},
            },
        },
        "required": ["assignments", "unsupported_scene_ids"],
        "additionalProperties": False,
    }


def _validate_attribution_response(
    raw: dict[str, Any],
    source_urls: list[str],
) -> list[int]:
    unsupported = raw.get("unsupported_scene_ids")
    if not isinstance(unsupported, list):
        raise local_llm.LocalLLMError(
            "Scene attribution response omitted unsupported_scene_ids"
        )
    if unsupported:
        scene_ids = sorted({int(value) for value in unsupported})
        raise _UnsupportedSceneAttribution(
            "Selected sources do not support scene(s): "
            + ", ".join(str(value) for value in scene_ids)
        )

    assignments = raw.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 6:
        raise local_llm.LocalLLMError(
            "Scene attribution response must contain exactly six assignments"
        )

    index_by_url = {url: index for index, url in enumerate(source_urls)}
    assigned: dict[int, int] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise local_llm.LocalLLMError(
                "Every scene attribution assignment must be an object"
            )
        scene_id = assignment.get("scene_id")
        if isinstance(scene_id, bool) or not isinstance(scene_id, int):
            raise local_llm.LocalLLMError(
                "Every scene attribution assignment requires an integer scene_id"
            )
        if not 0 <= scene_id < 6 or scene_id in assigned:
            raise local_llm.LocalLLMError(
                f"Invalid or duplicate scene attribution id: {scene_id}"
            )
        source_url = str(assignment.get("source_url") or "")
        if source_url not in index_by_url:
            raise local_llm.LocalLLMError(
                f"Scene attribution returned an unselected source URL: {source_url}"
            )
        assigned[scene_id] = index_by_url[source_url]

    if set(assigned) != set(range(6)):
        raise local_llm.LocalLLMError(
            "Scene attribution must assign every scene id from 0 through 5 exactly once"
        )
    return [assigned[scene_id] for scene_id in range(6)]


def _repair_scene_attribution(
    settings: Settings,
    scenes_raw: list[Any],
    source_urls: list[str],
    sources: list[SourceItem],
    *,
    attempts: int = 2,
) -> list[int]:
    """Map every scene to an exact selected URL, then convert URLs to indices.

    Numeric model indices are never clamped, shifted, wrapped, or guessed. The
    focused pass may select only URLs already validated by the package. Any
    unsupported scene or unknown URL fails closed before voice generation.
    """
    if not source_urls:
        raise local_llm.LocalLLMError("Scene attribution requires selected source URLs")

    scenes = _scene_records(scenes_raw)
    evidence = _selected_source_records(source_urls, sources)
    schema = _attribution_schema(source_urls)
    prompt = f"""
You are repairing evidence attribution only. Do not rewrite, expand, or reinterpret any scene.

For each of the six existing scenes, choose exactly one source_url copied verbatim from SELECTED EVIDENCE that directly supports the scene's factual claim. A source may support multiple scenes. Do not use numeric source positions, catalog identifiers, publisher names, or URLs outside SELECTED EVIDENCE.

When no selected source supports a scene, include that scene_id in unsupported_scene_ids. Never force an unsupported assignment.

Return one JSON object with:
- assignments: exactly six objects containing scene_id and source_url
- unsupported_scene_ids: an array, empty only when all six claims are supported

SCENES:
{json.dumps(scenes, ensure_ascii=False)}

SELECTED EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return one JSON object only. Do not include markdown, analysis, "
                    "or hidden reasoning. /no_think"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
        "stream": False,
        "response_format": {"type": "json_object", "schema": schema},
    }

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                f"{settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
            if response.status_code >= 400:
                raise local_llm.LocalLLMError(
                    "llama.cpp scene attribution returned "
                    f"{response.status_code}: {response.text[:1200]}"
                )
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise local_llm.LocalLLMError(
                    "Scene attribution response contained no choices"
                )
            content = ((choices[0].get("message") or {}).get("content"))
            if not isinstance(content, str) or not content.strip():
                raise local_llm.LocalLLMError(
                    "Scene attribution response contained no message content"
                )
            raw = local_llm._extract_json(content)
            return _validate_attribution_response(raw, source_urls)
        except _UnsupportedSceneAttribution:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)

    raise local_llm.LocalLLMError(
        f"Exact scene-source attribution failed after {attempts} attempts: {last_error}"
    ) from last_error


def _strict_scene_indices(
    scenes_raw: list[Any],
    source_urls: list[str],
    sources: list[SourceItem],
) -> list[int]:
    """Preserve integer mappings exactly; never infer one-based or catalog indices."""
    del source_urls, sources
    indices: list[int] = []
    for scene in scenes_raw:
        if not isinstance(scene, dict) or "source_index" not in scene:
            raise local_llm.LocalLLMError(
                "Every scene requires an integer source_index"
            )
        value = scene["source_index"]
        if isinstance(value, bool):
            raise local_llm.LocalLLMError(
                "Every scene requires an integer source_index"
            )
        try:
            indices.append(int(value))
        except (TypeError, ValueError) as exc:
            raise local_llm.LocalLLMError(
                "Every scene requires an integer source_index"
            ) from exc
    return indices


def _copy_with_indices(raw: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    repaired = json.loads(json.dumps(raw))
    scenes = repaired.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 6:
        raise local_llm.LocalLLMError(
            "Exact attribution repair requires exactly six scenes"
        )
    for scene, source_index in zip(scenes, indices, strict=True):
        if not isinstance(scene, dict):
            raise local_llm.LocalLLMError(
                "Exact attribution repair encountered a non-object scene"
            )
        scene["source_index"] = source_index
    return repaired


def _is_scene_index_failure(exc: Exception) -> bool:
    message = str(exc)
    return message.startswith("Scene source_index") or message.startswith(
        "Every scene requires an integer source_index"
    )


def generate_package(
    settings: Settings,
    sources: list[SourceItem],
    strategy: Strategy,
) -> VideoPackage:
    """Generate a package with strict integer and exact-URL attribution handling."""
    with _GENERATION_LOCK:
        original_package_from_raw = local_llm._package_from_raw
        original_normalizer = local_llm._normalize_scene_source_indices

        def package_with_exact_attribution(
            package_settings: Settings,
            package_sources: list[SourceItem],
            raw: dict[str, Any],
        ) -> VideoPackage:
            local_llm._normalize_scene_source_indices = _strict_scene_indices
            try:
                return original_package_from_raw(
                    package_settings,
                    package_sources,
                    raw,
                )
            except local_llm.LocalLLMError as exc:
                if not _is_scene_index_failure(exc):
                    raise
                scenes = raw.get("scenes")
                source_urls = raw.get("source_urls")
                if not isinstance(scenes, list) or not isinstance(source_urls, list):
                    raise
                indices = _repair_scene_attribution(
                    package_settings,
                    scenes,
                    [str(url) for url in source_urls],
                    package_sources,
                )
                repaired = _copy_with_indices(raw, indices)
                return original_package_from_raw(
                    package_settings,
                    package_sources,
                    repaired,
                )

        local_llm._package_from_raw = package_with_exact_attribution
        local_llm._normalize_scene_source_indices = _strict_scene_indices
        try:
            return local_llm.generate_package(settings, sources, strategy)
        finally:
            local_llm._package_from_raw = original_package_from_raw
            local_llm._normalize_scene_source_indices = original_normalizer
