from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .feeds import SourceItem
from .models import Scene, VideoPackage
from .production_narration_length import stabilize_production_narration


_INSTALLED = False
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{3,}", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "about", "after", "also", "allowing", "based", "being", "broader", "built",
    "change", "changes", "creating", "development", "developments", "different",
    "efficient", "efficiency", "emphasizing", "framework", "frameworks", "growing",
    "helping", "highlighted", "highlights", "improve", "improves", "innovation",
    "innovations", "learning", "models", "platform", "recent", "reports", "research",
    "source", "systems", "technology", "their", "these", "tools", "toward", "turning",
    "using", "while", "which", "with", "work", "world",
}


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value) if token.casefold() not in _STOP}


def _publisher_pattern(publisher: str) -> re.Pattern[str]:
    words = [re.escape(part) for part in publisher.split() if part]
    return re.compile(r"\b" + r"\s+".join(words) + r"(?:'s)?\b", re.IGNORECASE)


def _selected_sources(package: VideoPackage, sources: list[SourceItem]) -> list[SourceItem]:
    by_url = {source.url: source for source in sources}
    return [by_url[url] for url in package.source_urls if url in by_url]


def _title_owners(sources: Iterable[SourceItem]) -> dict[str, set[int]]:
    owners: dict[str, set[int]] = {}
    for index, source in enumerate(sources):
        publisher_tokens = _tokens(source.publisher)
        for token in _tokens(source.title) - publisher_tokens:
            owners.setdefault(token, set()).add(index)
    return owners


def _cross_attributed(sentence: str, sources: list[SourceItem], owners: dict[str, set[int]]) -> int | None:
    sentence_tokens = _tokens(sentence)
    for index, source in enumerate(sources):
        if not _publisher_pattern(source.publisher).search(sentence):
            continue
        own_evidence = _tokens(f"{source.title} {source.summary}")
        for token in sentence_tokens:
            token_owners = owners.get(token, set())
            if token_owners and index not in token_owners and token not in own_evidence:
                return index
    return None


def _source_sentence(source: SourceItem) -> str:
    summary_words = source.summary.split()
    summary = " ".join(summary_words[:28]).strip(" ,.;")
    title = " ".join(source.title.split()).strip(" ,.;")
    if summary:
        return f"{source.publisher} separately reports {title}. Its source describes {summary}."
    return f"{source.publisher} separately reports {title}."


def _repair_text(text: str, sources: list[SourceItem], owners: dict[str, set[int]]) -> tuple[str, int]:
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(text.strip()) if sentence.strip()]
    repaired: list[str] = []
    repairs = 0
    for sentence in sentences:
        source_index = _cross_attributed(sentence, sources, owners)
        if source_index is None:
            repaired.append(sentence)
            continue
        repaired.append(_source_sentence(sources[source_index]))
        repairs += 1
    return " ".join(repaired), repairs


def _repair_scene(
    scene: Scene,
    sources: list[SourceItem],
    owners: dict[str, set[int]],
) -> tuple[Scene, int]:
    combined = f"{scene.heading}. {scene.body}. {scene.visual}."
    offending = _cross_attributed(combined, sources, owners)
    if offending is None:
        return scene, 0
    source_index = scene.source_index if 0 <= scene.source_index < len(sources) else offending
    source = sources[source_index]
    heading = " ".join(source.title.split()[:8]).strip(" ,.;")[:72]
    return (
        replace(
            scene,
            heading=heading,
            body=_source_sentence(source),
            visual=(
                "A single continuous editorial photograph of the real-world subject described "
                "by this source, without publisher marks, product lettering, screens, panels, or text."
            ),
        ),
        1,
    )


def ground_cross_publisher_claims(
    package: VideoPackage,
    sources: list[SourceItem],
) -> VideoPackage:
    """Replace claims that assign one source's named subject to another publisher.

    Publishers and evidence URLs remain independent. The repair is deliberately narrow:
    it triggers only when a sentence names a selected publisher while also using a
    distinctive title token owned by a different selected source and absent from the
    named publisher's own title/summary evidence.
    """
    selected = _selected_sources(package, sources)
    if len(selected) < 2:
        return package
    owners = _title_owners(selected)

    narration, narration_repairs = _repair_text(package.narration, selected, owners)
    description, description_repairs = _repair_text(package.description, selected, owners)
    scenes: list[Scene] = []
    scene_repairs = 0
    for scene in package.scenes:
        repaired, count = _repair_scene(scene, selected, owners)
        scenes.append(repaired)
        scene_repairs += count

    if narration_repairs + description_repairs + scene_repairs == 0:
        return package
    narration_payload = stabilize_production_narration({"narration": narration})
    narration = str(narration_payload.get("narration") or narration)
    return replace(
        package,
        narration=narration,
        description=description,
        scenes=scenes,
    )


def install_production_claim_attribution() -> None:
    """Install cross-publisher claim ownership after relationship grounding."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_content

    original = production_content._ground_generic_copy

    def attributed(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
        return ground_cross_publisher_claims(original(package, sources), sources)

    production_content._ground_generic_copy = attributed
    _INSTALLED = True
