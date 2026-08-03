from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .feeds import SourceItem
from .models import VideoPackage


_INSTALLED = False
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]{2,}")
_STOPWORDS = {
    "about", "across", "after", "agent", "agents", "also", "artificial",
    "available", "build", "built", "company", "development", "developers",
    "digital", "from", "future", "google", "highlighted", "innovation",
    "intelligence", "latest", "managed", "microsoft", "model", "models",
    "more", "nvidia", "openai", "platform", "recent", "research", "system",
    "systems", "technology", "their", "these", "this", "tools", "using",
    "what", "with", "work",
}
_PUBLISHER_ALIASES = {
    "microsoft research": ("microsoft research", "microsoft"),
    "google ai": ("google ai", "google"),
    "openai": ("openai",),
    "nvidia": ("nvidia",),
    "anthropic": ("anthropic",),
    "hugging face": ("hugging face", "huggingface"),
}


class StoryCoherenceError(RuntimeError):
    pass


def _tokens(value: object) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(str(value))
        if token.casefold() not in _STOPWORDS and len(token) >= 4
    }


def _entity_tokens(source: SourceItem) -> set[str]:
    publisher_tokens = _tokens(source.publisher)
    title_tokens = _TOKEN_RE.findall(source.title)
    distinctive: set[str] = set()
    for token in title_tokens:
        folded = token.casefold()
        if folded in _STOPWORDS or folded in publisher_tokens or len(token) < 4:
            continue
        internal_upper = any(character.isupper() for character in token[1:])
        has_digit = any(character.isdigit() for character in token)
        is_acronym = token.isupper() and len(token) >= 3
        if internal_upper or has_digit or is_acronym or "-" in token:
            distinctive.add(folded)
    if distinctive:
        return distinctive
    candidates = sorted(
        _tokens(source.title) - publisher_tokens,
        key=lambda token: (-len(token), token),
    )
    return set(candidates[:4])


def _source_tokens(source: SourceItem) -> set[str]:
    return _tokens(f"{source.publisher} {source.title} {source.summary}")


def _selected_sources(package: VideoPackage, sources: list[SourceItem]) -> list[SourceItem]:
    by_url = {source.url: source for source in sources}
    selected: list[SourceItem] = []
    for url in package.source_urls:
        source = by_url.get(url)
        if source is None:
            raise StoryCoherenceError(f"Selected source is missing from evidence: {url}")
        selected.append(source)
    if not selected:
        raise StoryCoherenceError("Production package selected no primary source")
    return selected


def _publisher_mentions(text: str, selected: list[SourceItem]) -> set[int]:
    lowered = text.casefold()
    mentions: set[int] = set()
    for index, source in enumerate(selected):
        aliases = _PUBLISHER_ALIASES.get(
            source.publisher.casefold(),
            (source.publisher.casefold(),),
        )
        if any(alias and alias in lowered for alias in aliases):
            mentions.add(index)
    return mentions


def _validate_secondary_source_coherence(selected: list[SourceItem]) -> None:
    primary = selected[0]
    primary_entities = _entity_tokens(primary)
    primary_topic = _tokens(primary.title)
    for index, source in enumerate(selected[1:], start=1):
        tokens = _source_tokens(source)
        entity_overlap = primary_entities & tokens
        topic_overlap = primary_topic & _tokens(source.title)
        if not entity_overlap and len(topic_overlap) < 2:
            raise StoryCoherenceError(
                "Selected source is unrelated to the primary story: "
                f"source_index={index}, primary={primary.title!r}, secondary={source.title!r}"
            )


def _validate_scene_claims(package: VideoPackage, selected: list[SourceItem]) -> None:
    primary_entities = _entity_tokens(selected[0])
    for scene_index, scene in enumerate(package.scenes):
        source_index = int(scene.source_index)
        if not 0 <= source_index < len(selected):
            raise StoryCoherenceError(
                f"Scene {scene_index} source_index is outside selected evidence"
            )
        source = selected[source_index]
        evidence_tokens = _source_tokens(source)
        claim = f"{scene.heading} {scene.body}"
        claim_tokens = _tokens(claim)
        borrowed_entities = primary_entities & claim_tokens
        unsupported_entities = borrowed_entities - evidence_tokens
        if source_index != 0 and unsupported_entities:
            raise StoryCoherenceError(
                f"Scene {scene_index} attributes primary subject token(s) "
                f"{sorted(unsupported_entities)} to unrelated source_index={source_index}"
            )

        mentioned_publishers = _publisher_mentions(claim, selected)
        foreign_mentions = mentioned_publishers - {source_index}
        evidence_text = f"{source.title} {source.summary}".casefold()
        unsupported_publishers = {
            index
            for index in foreign_mentions
            if selected[index].publisher.casefold() not in evidence_text
        }
        if unsupported_publishers:
            names = [selected[index].publisher for index in sorted(unsupported_publishers)]
            raise StoryCoherenceError(
                f"Scene {scene_index} assigns unsupported publisher reference(s) {names} "
                f"to source_index={source_index}"
            )


def validate_story_coherence(
    package: VideoPackage,
    sources: list[SourceItem],
) -> None:
    selected = _selected_sources(package, sources)
    _validate_secondary_source_coherence(selected)
    _validate_scene_claims(package, selected)


def _thumbnail_entity(source: SourceItem) -> str:
    publisher_tokens = _tokens(source.publisher)
    raw_tokens = _TOKEN_RE.findall(source.title)
    for token in raw_tokens:
        folded = token.casefold()
        if folded in _STOPWORDS or folded in publisher_tokens or len(token) < 4:
            continue
        if any(character.isupper() for character in token[1:]) or any(
            character.isdigit() for character in token
        ):
            return token
    candidates = [
        token
        for token in raw_tokens
        if token.casefold() not in _STOPWORDS
        and token.casefold() not in publisher_tokens
        and len(token) >= 4
    ]
    return candidates[0] if candidates else source.publisher.split()[0]


def repair_thumbnail_copy(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
    selected = _selected_sources(package, sources)
    entity = _thumbnail_entity(selected[0]).strip(" -—:;,.!")
    thumbnail = f"{entity} EXPLAINED".upper()
    return replace(package, thumbnail_text=thumbnail[:45])


def install_production_story_coherence() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_content
    from .local_llm import LocalLLMError

    original_ground = production_content._ground_generic_copy
    original_validate = production_content._validate_publishable_content

    def ground(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
        return repair_thumbnail_copy(original_ground(package, sources), sources)

    def validate(package: VideoPackage, sources: list[SourceItem]) -> None:
        original_validate(package, sources)
        try:
            validate_story_coherence(package, sources)
        except StoryCoherenceError as exc:
            raise LocalLLMError(str(exc)) from exc

    production_content._ground_generic_copy = ground
    production_content._validate_publishable_content = validate
    _INSTALLED = True
