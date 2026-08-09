from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Sequence

from .feeds import SourceItem
from .models import VideoPackage


_INSTALLED = False


def focused_narration_prompt_v69(
    package: VideoPackage,
    sources: Sequence[SourceItem],
    validation_error: str,
) -> str:
    """Use a deliberately tiny response schema for the local 4B editorial repair model."""
    from .production_vimax_human_editorial_v66 import (
        _EDITORIAL_CONTRACT,
        _selected_editorial_evidence,
    )

    return f"""
You are the final human-style narration editor for a factual vertical technology short.
Return JSON only. Repair the spoken narration and nothing else.

The current narration failed:
{validation_error}

{_EDITORIAL_CONTRACT}

HARD BOUNDARY:
- Return exactly one field: narration.
- narration must contain 132-134 whitespace-separated words; count before returning.
- Preserve only concrete facts already supported by SELECTED EVIDENCE.
- Remove all internal sourcing/attribution-process language and generic filler.
- Keep the supported release actor correct.
- Do not add a number, benchmark, partner, capability, location, relationship, or result absent from the evidence.
- Do not explain your edit and do not return title, scenes, sources, notes, or markdown.
- If the evidence cannot support 132-134 useful words, return exactly {{"skip_reason":"specific reason"}}.

CURRENT NARRATION:
{package.narration}

SELECTED EVIDENCE:
{json.dumps(_selected_editorial_evidence(package, sources), ensure_ascii=False)}

Return exactly:
{{"narration":"..."}}
""".strip()


def apply_focused_narration_rewrite_v69(
    package: VideoPackage,
    raw: dict[str, Any],
) -> VideoPackage:
    from . import local_llm
    from .production_vimax_copy_integrity_v68 import normalize_finished_copy_v68

    if raw.get("skip_reason"):
        raise local_llm.LocalLLMError(f"Final narration rewrite declined: {raw['skip_reason']}")
    narration = normalize_finished_copy_v68(raw.get("narration"))
    if not narration:
        keys = sorted(str(key) for key in raw)
        raise local_llm.LocalLLMError(
            "Final narration rewrite returned no narration field; "
            f"response_keys={keys}"
        )
    unexpected = set(raw) - {"narration"}
    if unexpected:
        raise local_llm.LocalLLMError(
            "Final narration rewrite returned forbidden fields: "
            + ", ".join(sorted(str(key) for key in unexpected))
        )
    return replace(package, narration=narration)


def install_production_vimax_focused_copy_protocol_v69() -> None:
    """Use narration-only repair by default, but restore full copy repair when scenes failed.

    A narration-only schema is cheaper and more reliable for ordinary copy cleanup. It cannot repair
    viewer-facing scene text, so scene-copy failures deliberately fall back to v66's bounded full
    title/narration/scene rewrite instead of looping on an immutable bad scene.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_vimax_human_editorial_v66 as v66

    full_prompt = v66._focused_editorial_prompt_v66
    full_apply = v66._apply_focused_editorial_rewrite_v66

    def routed_prompt(
        package: VideoPackage,
        sources: Sequence[SourceItem],
        validation_error: str,
    ) -> str:
        if "scene copy" in str(validation_error).casefold():
            return full_prompt(package, sources, validation_error)
        return focused_narration_prompt_v69(package, sources, validation_error)

    def routed_apply(package: VideoPackage, raw: dict[str, Any]) -> VideoPackage:
        if "scenes" in raw or "title" in raw:
            return full_apply(package, raw)
        return apply_focused_narration_rewrite_v69(package, raw)

    v66._focused_editorial_prompt_v66 = routed_prompt
    v66._apply_focused_editorial_rewrite_v66 = routed_apply
    _INSTALLED = True
