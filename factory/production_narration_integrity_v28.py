from __future__ import annotations

import re
from typing import Any, Sequence

from .models import VideoPackage


_INSTALLED = False
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MALFORMED_TRANSITION_RE = re.compile(
    r"\b(?:fostering|enabling|supporting|driving|advancing|encouraging)\s+"
    r"(?:continued|ongoing|further)\s+[A-Z][^.?!]{8,}",
)
_ALLOWED_TITLE_INTRODUCERS = {
    "called", "named", "titled", "introduces", "introduced", "announced", "released",
}


def _tokens(value: object) -> list[str]:
    return _WORD_RE.findall(str(value).casefold())


def _near_title_window(sentence_tokens: list[str], title_tokens: list[str]) -> int | None:
    if len(title_tokens) < 5 or len(sentence_tokens) < len(title_tokens):
        return None
    width = len(title_tokens)
    for start in range(0, len(sentence_tokens) - width + 1):
        window = sentence_tokens[start : start + width]
        if window[0] != title_tokens[0]:
            continue
        positional_matches = sum(
            left == right for left, right in zip(window, title_tokens, strict=True)
        )
        overlap = len(set(window) & set(title_tokens)) / max(1, len(set(title_tokens)))
        if positional_matches / width >= 0.75 and overlap >= 0.75:
            return start
    return None


def validate_narration_integrity_v28(
    package: VideoPackage,
    sources: Sequence[Any],
) -> None:
    """Reject pasted source-title fragments and malformed sentence transitions."""
    from .local_llm import LocalLLMError

    narration = " ".join(package.narration.split())
    malformed = _MALFORMED_TRANSITION_RE.search(narration)
    if malformed:
        raise LocalLLMError(
            "Production narration contains a malformed transition into a pasted title fragment: "
            + malformed.group(0)[:120]
        )

    source_by_url = {str(getattr(source, "url", "")): source for source in sources}
    selected = [source_by_url[url] for url in package.source_urls if url in source_by_url]
    title_token_sets = [
        _tokens(getattr(source, "title", ""))
        for source in selected
        if len(_tokens(getattr(source, "title", ""))) >= 5
    ]
    if not title_token_sets:
        return

    for sentence in _SENTENCE_RE.split(narration):
        sentence_tokens = _tokens(sentence)
        for title_tokens in title_token_sets:
            start = _near_title_window(sentence_tokens, title_tokens)
            if start is None or start == 0:
                continue
            prefix = sentence_tokens[max(0, start - 4) : start]
            if any(token in _ALLOWED_TITLE_INTRODUCERS for token in prefix):
                continue
            raise LocalLLMError(
                "Production narration contains a source title pasted into the middle of a "
                "sentence instead of grammatical copy"
            )


def install_production_narration_integrity_v28() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_content

    original_validate = production_content._validate_publishable_content

    def validate(package: VideoPackage, sources: list[Any]) -> None:
        original_validate(package, sources)
        validate_narration_integrity_v28(package, sources)

    production_content._validate_publishable_content = validate
    _INSTALLED = True
