from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .feeds import SourceItem


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.-]{2,}")
_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "for",
    "from",
    "into",
    "its",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
}
_INSTALLED = False


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value).casefold())
        if token not in _STOP_WORDS
    }


def _selected_sources(
    source_urls: list[str], sources: list[SourceItem]
) -> list[SourceItem | None]:
    by_url: dict[str, SourceItem] = {}
    for source in sources:
        by_url.setdefault(source.url, source)
    return [by_url.get(url) for url in source_urls]


def _best_selected_source_index(
    scene: dict[str, Any],
    selected_sources: list[SourceItem | None],
    *,
    fallback: int,
) -> int:
    scene_tokens = _tokens(
        " ".join(
            str(scene.get(field) or "")
            for field in ("heading", "body", "visual")
        )
    )
    scores: list[tuple[int, int]] = []
    for index, source in enumerate(selected_sources):
        if source is None:
            scores.append((0, -index))
            continue
        title_score = len(scene_tokens & _tokens(source.title)) * 4
        publisher_score = len(scene_tokens & _tokens(source.publisher)) * 3
        summary_score = len(scene_tokens & _tokens(source.summary))
        scores.append((title_score + publisher_score + summary_score, -index))

    best_score, negative_index = max(scores, default=(0, -fallback))
    if best_score <= 0:
        return fallback
    return -negative_index


def _production_normalizer(
    original: Callable[[list[Any], list[str], list[SourceItem]], list[int]],
    scenes_raw: list[Any],
    source_urls: list[str],
    sources: list[SourceItem],
) -> list[int]:
    try:
        return original(scenes_raw, source_urls, sources)
    except Exception as exc:
        # Preserve all validation failures except source-index convention/range errors.
        message = str(exc)
        if "source_index" not in message:
            raise

    if not source_urls:
        raise ValueError("Cannot repair scene source indices without selected sources")

    raw_indices: list[int] = []
    typed_scenes: list[dict[str, Any]] = []
    for raw_scene in scenes_raw:
        if not isinstance(raw_scene, dict):
            raise ValueError("Every scene must be an object")
        value = raw_scene.get("source_index")
        if isinstance(value, bool):
            raise ValueError("Every scene requires an integer source_index")
        try:
            raw_indices.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Every scene requires an integer source_index") from exc
        typed_scenes.append(raw_scene)

    selected_sources = _selected_sources(source_urls, sources)
    scene_count = len(typed_scenes)
    ordinal_pattern = raw_indices in (
        list(range(scene_count)),
        list(range(1, scene_count + 1)),
    )

    repaired: list[int] = []
    for position, (scene, raw_index) in enumerate(zip(typed_scenes, raw_indices, strict=True)):
        if not ordinal_pattern and 0 <= raw_index < len(source_urls):
            repaired.append(raw_index)
            continue
        fallback = position % len(source_urls)
        repaired.append(
            _best_selected_source_index(
                scene,
                selected_sources,
                fallback=fallback,
            )
        )

    if not all(0 <= index < len(source_urls) for index in repaired):
        raise ValueError("Repaired scene source indices are outside selected sources")
    return repaired


def install_source_index_repair() -> None:
    """Install an idempotent production repair around local-LLM source mapping.

    The original strict validator always runs first. The fallback activates only for
    source-index convention/range failures and maps scenes exclusively to URLs that
    already passed the package's source and publisher validation.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original = local_llm._normalize_scene_source_indices

    def normalize(
        scenes_raw: list[Any],
        source_urls: list[str],
        sources: list[SourceItem],
    ) -> list[int]:
        return _production_normalizer(original, scenes_raw, source_urls, sources)

    local_llm._normalize_scene_source_indices = normalize
    _INSTALLED = True
