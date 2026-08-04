from __future__ import annotations

from .visual_storyboard_v30 import clean


_INSTALLED = False


def classify_claim_v30(claim: str) -> str:
    """Classify by the most specific production meaning, not the first generic keyword."""
    lowered = clean(claim).casefold()
    if any(term in lowered for term in ("before adoption", "controlled task", "test the claim", "repeatability", "failure rate")):
        return "controlled_test"
    if any(
        term in lowered
        for term in (
            "computing, data, software, and expertise",
            "computing data software and expertise",
            "expertise needed",
            "software and expertise",
            "providing computing",
        )
    ):
        return "expertise_support"
    if any(term in lowered for term in ("computing resources", "compute resources", "advanced computing", "data infrastructure")):
        return "compute_resources"
    if any(term in lowered for term in ("state and multistate", "nationwide", "across the us", "regional", "stronger foundation")):
        return "regional_network"
    if any(term in lowered for term in ("expanding access", "tools and knowledge", "accessible", "access to ai", "knowledge")):
        return "access_knowledge"
    if any(term in lowered for term in ("joining", "hubs program", "partnership", "collaboration", "initiative")):
        return "partnership_hub"
    if any(term in lowered for term in ("educator", "student", "education", "learning", "teach", "classroom")):
        return "education"
    if any(term in lowered for term in ("expertise", "support", "research")):
        return "expertise_support"
    return "partnership_hub"


def install_production_visual_storyboard_priority_v30() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import visual_storyboard_v30

    visual_storyboard_v30.classify_claim = classify_claim_v30
    _INSTALLED = True
