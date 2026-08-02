from __future__ import annotations

import re
from dataclasses import replace

from .feeds import SourceItem
from .models import VideoPackage
from .production_narration_length import stabilize_production_narration


_INSTALLED = False
_RELATIONSHIP_PATTERNS = (
    r"\bcollaboration\s+between\b",
    r"\bpartnership\s+between\b",
    r"\bjointly\s+(?:developed|released|built|created)\b",
    r"\bco-developed\b",
    r"\bworked\s+together\b",
    r"\bin\s+partnership\s+with\b",
)
_RELATIONSHIP_EVIDENCE_TERMS = (
    "collaboration",
    "collaborat",
    "partnership",
    "partnered",
    "jointly",
    "co-developed",
    "worked together",
    "in partnership with",
)


def _selected_sources(package: VideoPackage, sources: list[SourceItem]) -> list[SourceItem]:
    by_url = {source.url: source for source in sources}
    return [by_url[url] for url in package.source_urls if url in by_url]


def _declares_relationship(sources: list[SourceItem]) -> bool:
    evidence = " ".join(
        f"{source.publisher} {source.title} {source.summary}" for source in sources
    ).casefold()
    return any(term in evidence for term in _RELATIONSHIP_EVIDENCE_TERMS)


def _contains_relationship(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _RELATIONSHIP_PATTERNS)


def _publisher_names(sources: list[SourceItem]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for source in sources:
        name = source.publisher.strip()
        key = name.casefold()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def _neutral_context(names: list[str]) -> str:
    if len(names) >= 2:
        pair = f"{names[0]} and {names[1]}"
    elif names:
        pair = names[0]
    else:
        pair = "The selected publishers"
    return (
        f"{pair} provide separate primary-source context for this topic. "
        "Each report is evaluated independently and supports only its attributed claim."
    )


def _repair_sentences(text: str, replacement: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    repaired: list[str] = []
    inserted = False
    for sentence in sentences:
        if _contains_relationship(sentence):
            if not inserted:
                repaired.extend(
                    part.strip()
                    for part in re.split(r"(?<=[.!?])\s+", replacement)
                    if part.strip()
                )
                inserted = True
            continue
        repaired.append(sentence)
    return " ".join(repaired)


def ground_unsupported_relationships(
    package: VideoPackage,
    sources: list[SourceItem],
) -> VideoPackage:
    """Replace invented publisher relationships with explicit independent context."""
    selected = _selected_sources(package, sources)
    if not selected or _declares_relationship(selected):
        return package

    combined = " ".join(
        (
            package.topic,
            package.title,
            package.narration,
            package.description,
            package.thumbnail_text,
            package.top_comment,
            *(scene.heading for scene in package.scenes),
            *(scene.body for scene in package.scenes),
            *(scene.visual for scene in package.scenes),
        )
    )
    if not _contains_relationship(combined):
        return package

    names = _publisher_names(selected)
    neutral = _neutral_context(names)
    primary = selected[0]
    safe_title = f"{primary.publisher}: {' '.join(primary.title.split()[:7])}"[:78].rstrip(" -—:;,.")
    if len(safe_title) < 28:
        safe_title = f"{safe_title} — What Changed"[:78]

    narration = _repair_sentences(package.narration, neutral)
    stabilized = stabilize_production_narration({"narration": narration})
    narration = str(stabilized.get("narration") or narration)

    repaired_scenes = []
    for scene in package.scenes:
        if _contains_relationship(f"{scene.heading} {scene.body} {scene.visual}"):
            repaired_scenes.append(
                replace(
                    scene,
                    heading="Independent source context",
                    body="The selected reports are evaluated separately and support only their attributed claims.",
                    visual="Two distinct abstract evidence streams remain separate while entering one verification framework.",
                )
            )
        else:
            repaired_scenes.append(scene)

    description = _repair_sentences(package.description, neutral)
    top_comment = _repair_sentences(package.top_comment, neutral)
    return replace(
        package,
        topic=primary.title[:120],
        title=safe_title,
        narration=narration,
        description=description,
        top_comment=top_comment,
        scenes=repaired_scenes,
    )


def install_production_relationship_grounding() -> None:
    """Install deterministic source-relationship repair before editorial validation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_content

    original = production_content._ground_generic_copy

    def grounded(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
        return ground_unsupported_relationships(original(package, sources), sources)

    production_content._ground_generic_copy = grounded
    _INSTALLED = True
