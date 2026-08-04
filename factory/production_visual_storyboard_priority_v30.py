from __future__ import annotations

from .visual_storyboard_v30 import clean


_INSTALLED = False


def classify_claim_v30(claim: str) -> str:
    """Route a spoken claim by its most specific physical story, not generic AI vocabulary."""
    lowered = clean(claim).casefold()

    # Verification and evaluation claims must become measurable physical tests.
    if any(
        term in lowered
        for term in (
            "before adoption",
            "controlled task",
            "test the claim",
            "repeatability",
            "failure rate",
            "train and evaluate",
            "evaluate ai agents",
            "evaluation across",
            "sources:",
            "source:",
        )
    ):
        return "controlled_test"

    # A complete compute/data/software/expertise bundle is a human-support story, not merely racks.
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

    # Reuse, shared infrastructure, and model-performance claims require visible compute plumbing.
    if any(
        term in lowered
        for term in (
            "computing resources",
            "compute resources",
            "advanced computing",
            "data infrastructure",
            "reuse of infrastructure",
            "reuse infrastructure",
            "infrastructure reuse",
            "shared infrastructure",
            "smaller models",
            "strong performance",
        )
    ):
        return "compute_resources"

    # Explicit participation describes a partnership even when the programme name contains
    # regional, research, or education vocabulary.
    if any(term in lowered for term in ("joining", "hubs program")):
        return "partnership_hub"

    # Scale and multi-system orchestration are represented by a connected physical network.
    if any(
        term in lowered
        for term in (
            "state and multistate",
            "nationwide",
            "across the us",
            "regional",
            "stronger foundation",
            "scalable agentic",
            "complex ai systems",
            "multiple ai tasks",
            "wide range of tasks",
        )
    ):
        return "regional_network"

    # Explicit audience roles are more specific than generic accessibility language.
    if any(term in lowered for term in ("educator", "student", "education", "learning", "teach", "classroom")):
        return "education"

    # Access is a concrete outcome and must outrank generic partnership/support words.
    if any(term in lowered for term in ("expanding access", "tools and knowledge", "accessible", "access to ai", "knowledge")):
        return "access_knowledge"

    # Public/open availability and collaboration become a shared physical workspace.
    if any(
        term in lowered
        for term in (
            "public use",
            "open-source",
            "open source",
            "collaboration",
            "partnership",
            "initiative",
        )
    ):
        return "partnership_hub"

    # Efficiency and innovation claims require visible expert work rather than empty architecture.
    if any(
        term in lowered
        for term in (
            "efficient and effective",
            "focus on innovation",
            "rather than infrastructure",
            "expertise",
            "support",
            "research",
        )
    ):
        return "expertise_support"

    return "partnership_hub"


def install_production_visual_storyboard_priority_v30() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import visual_storyboard_v30

    visual_storyboard_v30.classify_claim = classify_claim_v30
    _INSTALLED = True
