from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import Settings
from .feeds import SourceItem
from .models import VideoPackage
from .policy import Strategy


_INSTALLED = False
_MAX_CANDIDATES = 4


class SingleStorySelectionError(RuntimeError):
    pass


def _concrete_title_score(source: SourceItem) -> int:
    """Prefer named products/releases without overriding live-trend ordering."""
    title = source.title.strip()
    score = 0
    for token in title.split():
        clean = token.strip("()[]{}:;,.!?—–-/")
        if not clean:
            continue
        if any(character.isdigit() for character in clean):
            score += 2
        if any(character.isupper() for character in clean[1:]):
            score += 2
        if "-" in clean or "." in clean:
            score += 1
    if ":" in title:
        score += 1
    return score


def select_story_candidates(
    sources: list[SourceItem],
    *,
    limit: int = _MAX_CANDIDATES,
) -> tuple[SourceItem, ...]:
    """Return trend-ranked authoritative candidates without mixing their evidence.

    Input order remains the dominant signal because trend_ranking already placed current
    demand first. Concrete-title score only breaks near-position ties, preventing generic
    lifestyle articles from outranking named AI products or releases.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not sources:
        raise SingleStorySelectionError("Production source selection received no articles")

    indexed = list(enumerate(sources[: max(limit * 2, limit)]))
    ranked = sorted(
        indexed,
        key=lambda item: (
            item[0] // 2,
            -_concrete_title_score(item[1]),
            item[0],
        ),
    )
    return tuple(source for _index, source in ranked[:limit])


def rewrite_prompt_for_single_authority(prompt: str) -> str:
    """Compile the generic package prompt into the production evidence contract."""
    rewritten = prompt.replace(
        "Select one current AI development that can be responsibly explained using at least 1 DISTINCT supplied publishers. A second publisher may provide context rather than independent confirmation, but do not imply independent confirmation when it is not present.",
        "Explain the ONE supplied authoritative article as a single coherent current AI story. Do not add another announcement, company, product, publisher, or implied confirmation.",
    )
    rewritten = rewritten.replace(
        "source_urls: 2-5 UNIQUE URLs copied exactly from the supplied entries and spanning at least 1 distinct publishers",
        "source_urls: exactly 1 UNIQUE URL copied exactly from the supplied entry",
    )
    rewritten = rewritten.replace(
        "- Choose source_urls from at least 1 different rows in PUBLISHER SOURCE OPTIONS.\n- Multiple URLs from one publisher still count as one publisher.",
        "- Choose exactly the one supplied URL.\n- Every factual claim and all six scenes must be supported by that article.",
    )
    rewritten = rewritten.replace(
        "If the sources cannot support one coherent package across that many publishers, return skip_reason rather than weakening attribution.",
        "If the supplied article cannot support six distinct factual scenes, return skip_reason rather than adding outside facts.",
    )
    return rewritten


def install_production_single_story_selection() -> None:
    """Try trend-ranked official articles independently until one passes all gates."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm, source_attributed_llm

    original_chat = local_llm._chat
    original_generate = source_attributed_llm.generate_package

    def single_story_chat(
        settings: Any,
        prompt: str,
        *,
        attempts: int = 3,
    ) -> dict[str, Any]:
        return original_chat(
            settings,
            rewrite_prompt_for_single_authority(prompt),
            attempts=attempts,
        )

    def single_story_generate(
        settings: Settings,
        sources: list[SourceItem],
        strategy: Strategy,
    ) -> VideoPackage:
        candidates = select_story_candidates(sources)
        errors: list[str] = []
        single_source_settings = replace(settings, min_primary_sources=1)
        for source in candidates:
            try:
                return original_generate(single_source_settings, [source], strategy)
            except local_llm.LocalLLMError as exc:
                errors.append(f"{source.title}: {exc}")
        raise local_llm.LocalLLMError(
            "No trend-ranked authoritative article produced a coherent package after "
            f"{len(candidates)} candidates: " + " | ".join(errors)
        )

    local_llm._chat = single_story_chat
    source_attributed_llm.generate_package = single_story_generate
    _INSTALLED = True
