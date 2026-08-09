from __future__ import annotations

from .feeds import SourceItem
from .models import VideoPackage


_INSTALLED = False


def ground_unsupported_relationships(
    package: VideoPackage,
    sources: list[SourceItem],
) -> VideoPackage:
    """Leave unsupported relationship claims visible for the strict validator.

    Previous deterministic repair replaced unsupported claims with audience-facing provenance
    boilerplate such as "Independent source context". That hid the factual defect and leaked
    internal source-evaluation language into scenes. The bounded production-content regeneration
    loop now owns the correction; this wrapper intentionally performs no textual substitution.
    """
    del sources
    return package


def install_production_relationship_grounding() -> None:
    """Keep the historical wrapper point without mutating unsupported viewer copy."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_content

    original = production_content._ground_generic_copy

    def grounded(package: VideoPackage, sources: list[SourceItem]) -> VideoPackage:
        return ground_unsupported_relationships(original(package, sources), sources)

    production_content._ground_generic_copy = grounded
    _INSTALLED = True
