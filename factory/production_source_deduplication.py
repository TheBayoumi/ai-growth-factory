from __future__ import annotations

import json
from typing import Any


_INSTALLED = False


def deduplicate_package_sources(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve first URL occurrence and force exact scene re-attribution.

    The model occasionally repeats the same selected URL while still returning a parallel
    publisher array and numeric scene positions. Removing a duplicate changes every later
    numeric position, so those numbers are intentionally invalidated. The existing exact-URL
    attribution pass then assigns all six claims against the de-duplicated selected evidence.
    No source is invented and no minimum-source or publisher-diversity gate is weakened.
    """
    urls = raw.get("source_urls")
    publishers = raw.get("source_publishers")
    if not isinstance(urls, list) or not isinstance(publishers, list):
        return raw
    if len(urls) != len(publishers):
        return raw

    normalized_urls = [str(value).strip() for value in urls]
    if len(set(normalized_urls)) == len(normalized_urls):
        return raw

    repaired = json.loads(json.dumps(raw))
    unique_urls: list[str] = []
    unique_publishers: list[str] = []
    seen: set[str] = set()
    for url, publisher in zip(normalized_urls, publishers, strict=True):
        if url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)
        unique_publishers.append(str(publisher).strip())

    repaired["source_urls"] = unique_urls
    repaired["source_publishers"] = unique_publishers
    scenes = repaired.get("scenes")
    if isinstance(scenes, list):
        for scene in scenes:
            if isinstance(scene, dict):
                scene["source_index"] = -1
    return repaired


def install_production_source_deduplication() -> None:
    """Normalize package JSON returned by the production chat boundary."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original_chat = local_llm._chat

    def deduplicating_chat(
        settings: Any,
        prompt: str,
        *,
        attempts: int = 3,
    ) -> dict[str, Any]:
        return deduplicate_package_sources(
            original_chat(settings, prompt, attempts=attempts)
        )

    local_llm._chat = deduplicating_chat
    _INSTALLED = True
