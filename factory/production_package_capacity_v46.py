from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import Any

from .feeds import SourceItem
from .models import Scene, VideoPackage


_INSTALLED = False
_SCENE_BODY_MAX_WORDS = 18
_SCENE_BODY_MAX_REPAIRABLE_WORDS = 30
_NARRATION_MIN_WORDS = 130
_NARRATION_MAX_WORDS = 140
_NARRATION_TARGET_WORDS = 136
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|x|×)?", re.IGNORECASE)
_CLAUSE_MARKERS = (
    " which ",
    " while ",
    " because ",
    " so that ",
    " but ",
    " and then ",
)
_CAVEAT_TERMS = (
    "before adopting",
    "but",
    "caveat",
    "compare",
    "evidence",
    "however",
    "limit",
    "test",
    "verify",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _finish_sentence(text: str) -> str:
    clean = text.rstrip(" ,;:—-")
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


def _compact_scene_body(text: str) -> str:
    """Compact a modest overshoot without character slicing or new claims."""
    normalized = " ".join(text.split())
    count = _word_count(normalized)
    if count <= _SCENE_BODY_MAX_WORDS:
        return normalized
    if count > _SCENE_BODY_MAX_REPAIRABLE_WORDS:
        return normalized

    lowered = normalized.casefold()
    for marker in _CLAUSE_MARKERS:
        position = lowered.find(marker)
        if position < 0:
            continue
        prefix = normalized[:position].strip()
        if 8 <= _word_count(prefix) <= _SCENE_BODY_MAX_WORDS:
            return _finish_sentence(prefix)

    for separator in (",", ";", "—", ":"):
        prefix = normalized.split(separator, 1)[0].strip()
        if 8 <= _word_count(prefix) <= _SCENE_BODY_MAX_WORDS:
            return _finish_sentence(prefix)

    return _finish_sentence(
        " ".join(normalized.split()[:_SCENE_BODY_MAX_WORDS])
    )


def stabilize_raw_scene_capacity(raw: dict[str, Any]) -> dict[str, Any]:
    scenes = raw.get("scenes")
    if not isinstance(scenes, list):
        return raw

    changed = False
    corrected_scenes: list[Any] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            corrected_scenes.append(scene)
            continue
        body = str(scene.get("body") or "")
        corrected_body = _compact_scene_body(body)
        if corrected_body == body:
            corrected_scenes.append(scene)
            continue
        corrected_scene = dict(scene)
        corrected_scene["body"] = corrected_body
        corrected_scenes.append(corrected_scene)
        changed = True

    if not changed:
        return raw
    corrected = dict(raw)
    corrected["scenes"] = corrected_scenes
    return corrected


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _selected_source_tokens(
    package: VideoPackage,
    sources: list[SourceItem],
) -> set[str]:
    selected_urls = set(package.source_urls)
    evidence = " ".join(
        f"{source.publisher} {source.author} {source.authority} "
        f"{source.title} {source.summary}"
        for source in sources
        if source.url in selected_urls
    )
    return {token.casefold() for token in _TOKEN_RE.findall(evidence)}


def _information_score(sentence: str, source_tokens: set[str]) -> int:
    tokens = {token.casefold() for token in _TOKEN_RE.findall(sentence)}
    source_hits = len(tokens & source_tokens)
    measurements = len(_NUMBER_RE.findall(sentence))
    lowered = sentence.casefold()
    caveats = sum(term in lowered for term in _CAVEAT_TERMS)
    return 1 + source_hits * 8 + measurements * 12 + caveats * 4


def _bounded_sentence_compression(
    narration: str,
    source_tokens: set[str],
) -> str:
    total = _word_count(narration)
    if total <= _NARRATION_MAX_WORDS:
        return narration

    sentences = _split_sentences(narration)
    if len(sentences) < 3:
        return _finish_sentence(
            " ".join(narration.split()[:_NARRATION_MAX_WORDS])
        )

    middle = sentences[1:-1]
    minimum_removal = total - _NARRATION_MAX_WORDS
    maximum_removal = total - _NARRATION_MIN_WORDS
    target_removal = total - _NARRATION_TARGET_WORDS
    candidates: list[tuple[tuple[int, int, int], set[int]]] = []

    if len(middle) <= 18:
        for size in range(1, len(middle) + 1):
            for indices in itertools.combinations(range(len(middle)), size):
                removed_words = sum(_word_count(middle[index]) for index in indices)
                if not minimum_removal <= removed_words <= maximum_removal:
                    continue
                removal_cost = sum(
                    _information_score(middle[index], source_tokens)
                    for index in indices
                )
                candidates.append(
                    (
                        (
                            removal_cost,
                            abs(removed_words - target_removal),
                            size,
                        ),
                        set(indices),
                    )
                )

    if candidates:
        _, removed = min(candidates, key=lambda item: item[0])
        corrected = " ".join(
            (
                sentences[0],
                *(
                    sentence
                    for index, sentence in enumerate(middle)
                    if index not in removed
                ),
                sentences[-1],
            )
        )
        if _NARRATION_MIN_WORDS <= _word_count(corrected) <= _NARRATION_MAX_WORDS:
            return corrected

    ranked = sorted(
        range(len(middle)),
        key=lambda index: (
            _information_score(middle[index], source_tokens),
            _word_count(middle[index]),
        ),
    )
    removed: set[int] = set()
    for index in ranked:
        proposed = removed | {index}
        final_count = total - sum(_word_count(middle[item]) for item in proposed)
        if final_count < _NARRATION_MIN_WORDS:
            continue
        removed = proposed
        if final_count <= _NARRATION_MAX_WORDS:
            corrected = " ".join(
                (
                    sentences[0],
                    *(
                        sentence
                        for item, sentence in enumerate(middle)
                        if item not in removed
                    ),
                    sentences[-1],
                )
            )
            if _NARRATION_MIN_WORDS <= _word_count(corrected) <= _NARRATION_MAX_WORDS:
                return corrected

    closing = sentences[-1]
    closing_words = _word_count(closing)
    prefix_budget = _NARRATION_MAX_WORDS - closing_words
    prefix = _finish_sentence(
        " ".join(" ".join(sentences[:-1]).split()[:prefix_budget])
    )
    corrected = f"{prefix} {closing}".strip()
    if _word_count(corrected) > _NARRATION_MAX_WORDS:
        corrected = _finish_sentence(
            " ".join(corrected.split()[:_NARRATION_MAX_WORDS])
        )
    return corrected


def stabilize_package_capacity(
    package: VideoPackage,
    sources: list[SourceItem],
) -> VideoPackage:
    source_tokens = _selected_source_tokens(package, sources)
    narration = _bounded_sentence_compression(package.narration, source_tokens)
    scenes = [
        replace(scene, body=_compact_scene_body(scene.body))
        for scene in package.scenes
    ]
    if narration == package.narration and all(
        corrected == original
        for corrected, original in zip(scenes, package.scenes, strict=True)
    ):
        return package
    return replace(package, narration=narration, scenes=scenes)


def _augment_repair_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nPRODUCTION CAPACITY REPAIR:\n"
        + "- Rewrite narration to 132-138 whitespace-separated words; count before returning.\n"
        + "- Every scene heading must contain at most 5 words.\n"
        + "- Every scene body must contain at most 18 words; count each body before returning.\n"
        + "- Preserve source URLs, source indices, measurements, named subjects, and factual meaning.\n"
    )


def install_production_package_capacity_v46() -> None:
    """Align generation and deterministic convergence with the frozen editorial limits."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm, production_content

    local_llm.NARRATION_TARGET_MIN_WORDS = 132
    local_llm.NARRATION_TARGET_MAX_WORDS = 138

    original_repair_prompt = local_llm._repair_prompt
    original_package_from_raw = local_llm._package_from_raw
    original_ground = production_content._ground_generic_copy

    def repair_prompt(*args: Any, **kwargs: Any) -> str:
        return _augment_repair_prompt(original_repair_prompt(*args, **kwargs))

    def package_from_raw(
        settings: Any,
        sources: list[SourceItem],
        raw: dict[str, Any],
    ) -> VideoPackage:
        return original_package_from_raw(
            settings,
            sources,
            stabilize_raw_scene_capacity(raw),
        )

    def ground(
        package: VideoPackage,
        sources: list[SourceItem],
    ) -> VideoPackage:
        return stabilize_package_capacity(original_ground(package, sources), sources)

    local_llm._repair_prompt = repair_prompt
    local_llm._package_from_raw = package_from_raw
    production_content._ground_generic_copy = ground
    _INSTALLED = True
