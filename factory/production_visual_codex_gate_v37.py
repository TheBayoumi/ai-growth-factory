from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import production_visual_codex_gate_v36 as codex_v36
from . import production_visual_semantic_review_v28 as semantic_v28
from .visual_storyboard_v30 import clean


_INSTALLED = False


def _semantic_key(value: str) -> str:
    return " ".join(word.casefold() for word in codex_v36._words(value))


def _contains_phrase(container: str, phrase: str) -> bool:
    container_key = _semantic_key(container)
    phrase_key = _semantic_key(phrase)
    if not phrase_key:
        return False
    return f" {phrase_key} " in f" {container_key} "


def validate_codex_visual_gate_v37(
    contract: codex_v36.SceneContractV36,
    compiled_prompt: str,
    negative_prompt: str,
) -> None:
    """Validate token-normalized phrases so punctuation changes do not create false failures."""
    missing = [
        phrase
        for phrase in contract.required_phrases
        if not _contains_phrase(compiled_prompt, phrase)
    ]
    conflicts = [
        phrase
        for phrase in contract.required_phrases
        if _contains_phrase(negative_prompt, phrase)
    ]
    substitutions = [
        phrase
        for phrase in contract.forbidden_substitutions
        if _contains_phrase(compiled_prompt, phrase)
    ]
    if missing or conflicts or substitutions:
        raise codex_v36.CodexVisualGateError(
            f"Scene contract {contract.identity} failed: missing={missing}; "
            f"negative_conflicts={conflicts}; forbidden_positive={substitutions}"
        )


def enrich_visual_review_v37(review: Any) -> Any:
    """Preserve the reviewer reason so regeneration can remove observed substitutions."""
    if str(getattr(review, "decision", "")).casefold() != "retry":
        return review
    reason = clean(str(getattr(review, "reason", "")))
    repair = clean(str(getattr(review, "repair_instruction", "")))
    combined = clean(". ".join(value for value in (reason, repair) if value))
    if not combined or combined == repair:
        return review
    return replace(review, repair_instruction=combined)


def install_production_visual_codex_gate_v37() -> None:
    """Install punctuation-safe contracts and stateful reviewer feedback propagation."""
    global _INSTALLED
    if _INSTALLED:
        return

    codex_v36.install_production_visual_codex_gate_v36()
    codex_v36.validate_codex_visual_gate_v36 = validate_codex_visual_gate_v37

    base_reviewer = semantic_v28.SemanticVisualReviewerV28

    class CodexStoryboardReviewerV37(base_reviewer):
        def review(self, *args: Any, **kwargs: Any) -> Any:
            return enrich_visual_review_v37(super().review(*args, **kwargs))

    CodexStoryboardReviewerV37.__name__ = "CodexStoryboardReviewerV37"
    semantic_v28.SemanticVisualReviewerV28 = CodexStoryboardReviewerV37
    _INSTALLED = True
