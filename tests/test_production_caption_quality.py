from __future__ import annotations

from types import SimpleNamespace

from factory.production_caption_quality import karaoke_text_safe, phrase_chunks_safe


def test_phrase_chunks_respect_word_and_character_limits() -> None:
    chunks = phrase_chunks_safe(
        "Microsoft Research turns repeated experience into evolving reusable knowledge for deployed agents."
    )
    assert chunks
    assert all(len(chunk.split()) <= 4 for chunk in chunks)
    assert all(len(chunk) <= 26 or len(chunk.split()) == 1 for chunk in chunks)
    assert " ".join(chunks) == (
        "Microsoft Research turns repeated experience into evolving reusable knowledge for deployed agents."
    )


def test_long_caption_uses_explicit_two_line_break() -> None:
    cue = SimpleNamespace(
        text="experience becomes evolving knowledge",
        start_seconds=0.0,
        end_seconds=2.4,
    )
    rendered = karaoke_text_safe(cue, lambda value: value)
    assert r"\N" in rendered
    assert "experience" in rendered
    assert "knowledge" in rendered
    assert rendered.count(r"\kf") == 4
