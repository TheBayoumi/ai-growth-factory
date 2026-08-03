from __future__ import annotations

import re
from typing import Any


_INSTALLED = False
_MAX_WORDS = 4
_MAX_CHARACTERS = 26


def phrase_chunks_safe(
    text: str,
    *,
    maximum_words: int = _MAX_WORDS,
    maximum_characters: int = _MAX_CHARACTERS,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = [*current, word]
        candidate_text = " ".join(candidate)
        punctuation_break = bool(re.search(r"[,;:!?]$", word)) and len(candidate) >= 2
        exceeds = len(candidate) > maximum_words or len(candidate_text) > maximum_characters
        if exceeds and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current = candidate
        if punctuation_break and current:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def _balanced_break(words: list[str]) -> int | None:
    if len(words) < 3:
        return None
    best_index: int | None = None
    best_delta: int | None = None
    for index in range(1, len(words)):
        left = len(" ".join(words[:index]))
        right = len(" ".join(words[index:]))
        delta = abs(left - right)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_index = index
    return best_index


def karaoke_text_safe(cue: Any, escape_ass: Any) -> str:
    words = cue.text.split()
    duration_cs = max(1, int(round((cue.end_seconds - cue.start_seconds) * 100)))
    base, remainder = divmod(duration_cs, len(words))
    break_index = _balanced_break(words) if len(cue.text) > 22 else None
    pieces: list[str] = []
    for index, word in enumerate(words):
        if break_index is not None and index == break_index:
            pieces.append(r"\N")
        word_duration = base + (1 if index < remainder else 0)
        pieces.append(rf"{{\kf{max(1, word_duration)}}}{escape_ass(word)}")
    return " ".join(pieces).replace(r"\N ", r"\N")


def install_production_caption_quality() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import caption_renderer

    def chunks(text: str, *, maximum_words: int = 6) -> list[str]:
        del maximum_words
        return phrase_chunks_safe(text)

    def karaoke(cue: Any) -> str:
        return karaoke_text_safe(cue, caption_renderer._escape_ass)

    caption_renderer._phrase_chunks = chunks
    caption_renderer._karaoke_text = karaoke
    _INSTALLED = True
