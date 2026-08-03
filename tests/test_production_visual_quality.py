from __future__ import annotations

from factory.production_visual_quality import strengthen_compiled_prompt
from factory.visual_prompt_compiler import CompiledVisualPrompt


def test_strengthened_prompt_blocks_v14_failure_modes() -> None:
    raw = CompiledVisualPrompt(
        director_prompt="Google AI dashboard poster with a woman holding a phone",
        compiled_prompt=(
            "Text-free cinematic editorial image. Google AI dashboard poster with a woman "
            "holding a phone. Subject high in frame. Dark empty lower third reserved for captions."
        ),
        negative_prompt="text, logo",
        word_count=18,
        word_budget=44,
    )
    result = strengthen_compiled_prompt(raw)
    lowered = result.compiled_prompt.casefold()
    assert "photorealistic conceptual scene" in lowered
    assert "no people" in lowered
    assert "google" not in lowered
    assert "woman" not in lowered
    assert "phone" not in lowered
    assert "dashboard" not in lowered
    assert "poster" not in lowered
    assert "collage" in result.negative_prompt.casefold()
    assert "pseudo-text" in result.negative_prompt.casefold()


def test_strengthened_prompt_preserves_concrete_mechanism_language() -> None:
    raw = CompiledVisualPrompt(
        director_prompt="Directional memory handoff",
        compiled_prompt=(
            "Text-free cinematic editorial image. A sequence of physical memory blocks moves "
            "through a directional handoff into a stable archive. Subject high in frame. "
            "Dark empty lower third reserved for captions."
        ),
        negative_prompt="text",
        word_count=29,
        word_budget=44,
    )
    result = strengthen_compiled_prompt(raw)
    assert "memory blocks" in result.compiled_prompt.casefold()
    assert "directional handoff" in result.compiled_prompt.casefold()
    assert result.compiler_version.endswith("+quality-v1")
