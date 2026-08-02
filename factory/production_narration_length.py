from __future__ import annotations

import re
from typing import Any


_INSTALLED = False
_MIN_REPAIRABLE_WORDS = 80
_TARGET_MIN_WORDS = 135
_TARGET_MAX_WORDS = 155
_SAFE_EXPANSIONS = (
    "Before adopting it, open the linked primary sources, reproduce the claim on a controlled task, and compare the result with the current workflow.",
    "Track latency, failure rate, human corrections, and repeatability so the decision follows measured behavior rather than a polished announcement.",
    "The evidence should determine the next step, not the headline alone.",
    "Record the result before changing the production system.",
)
_CTA_TERMS = (
    "subscribe",
    "follow",
    "comment",
    "link",
    "watch",
    "tell us",
    "share",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _insert_before_closing(sentences: list[str], additions: list[str]) -> list[str]:
    if not sentences:
        return additions
    closing = sentences[-1].casefold()
    if any(term in closing for term in _CTA_TERMS):
        return [*sentences[:-1], *additions, sentences[-1]]
    return [*sentences, *additions]


def stabilize_production_narration(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring a near-complete script into the production range without new claims.

    The finalizer adds only evidence-evaluation guidance: read the linked sources,
    reproduce the claim, record measurable behavior, and compare it with the current
    workflow. It never adds product capabilities, dates, benchmarks, relationships,
    pricing, or availability. Drafts below 80 words remain untouched and therefore fail
    closed because too much substantive content would have to be invented.
    """
    narration = str(raw.get("narration") or "").strip()
    count = _word_count(narration)
    if not _MIN_REPAIRABLE_WORDS <= count < 130:
        return raw

    additions: list[str] = []
    lowered = narration.casefold()
    running_count = count
    for sentence in _SAFE_EXPANSIONS:
        if sentence.casefold() in lowered:
            continue
        candidate_count = running_count + _word_count(sentence)
        if candidate_count > _TARGET_MAX_WORDS:
            continue
        additions.append(sentence)
        running_count = candidate_count
        if running_count >= _TARGET_MIN_WORDS:
            break

    if running_count < _TARGET_MIN_WORDS:
        return raw

    corrected = dict(raw)
    sentences = _split_sentences(narration)
    corrected["narration"] = " ".join(_insert_before_closing(sentences, additions))
    final_count = _word_count(str(corrected["narration"]))
    if not _TARGET_MIN_WORDS <= final_count <= _TARGET_MAX_WORDS:
        return raw
    return corrected


def install_production_narration_length_repair() -> None:
    """Install deterministic third-attempt narration stabilization."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original = local_llm._stabilize_near_minimum_narration

    def production_stabilizer(raw: dict[str, Any]) -> dict[str, Any]:
        corrected = stabilize_production_narration(raw)
        if corrected is not raw:
            return corrected
        return original(raw)

    local_llm._stabilize_near_minimum_narration = production_stabilizer
    _INSTALLED = True
