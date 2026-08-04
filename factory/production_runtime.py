from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production-only content, source, voice, visual, render, and QC policies.

    The production settings adapter is installed first so one authoritative source can be
    represented consistently from environment parsing through canary and publishing gates.
    Voice normalization uses strict peak headroom before any voice module binds the audio
    helpers, and deterministic QC evidence overrides stale reviewer feedback. Publisher
    names are canonicalized from selected evidence URLs before package validation. Scene
    attribution is then handled by factory.source_attributed_llm at the exact package
    boundary. Duplicate selected URLs are normalized at the production chat boundary and
    force exact URL re-attribution instead of preserving stale numeric positions.
    The production controller then tries trend-ranked authoritative articles independently;
    one official article supplies all factual claims while the separate trend snapshot proves
    current demand. This prevents unrelated announcements from being spliced merely to meet
    a publisher count. Near-complete narration length is stabilized before the editorial
    wrapper. Unsupported publisher relationships are grounded in explicit independent-source
    language before the same validator runs. A separate story-coherence gate rejects
    unrelated secondary sources and cross-source claim borrowing. Incomplete but otherwise
    valid Qwen Omni feedback is normalized before the bounded selective repair loop.
    Executable visual prompts must stay semantically distinct, generated keyframes pass an
    agentic image review/regeneration loop, and the final production compiler emits short
    object-only positive prompts that remain safely below the SDXL CLIP token limit. Before
    review, the complete lower 32% is replaced by subject-free negative space with a soft
    transition from the untouched upper image. Captions remain a separate animated ASS layer.
    Temporal QC excludes that caption layer while retaining strict generated-video checks.
    Finally, scene headings, bodies, and visual directions are validated before the legacy
    package parser can silently slice them, forcing the existing repair loop to return complete
    metadata rather than corrupted partial words.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_audio_qc import install_production_audio_qc
    from .production_caption_quality import install_production_caption_quality
    from .production_caption_zone import install_production_caption_zone
    from .production_content import install_production_content_gate
    from .production_narration_length import install_production_narration_length_repair
    from .production_object_visuals import install_production_object_visuals
    from .production_pacing import install_production_pacing
    from .production_relationship_grounding import install_production_relationship_grounding
    from .production_renderer import install_production_renderer
    from .production_reviewer_feedback import install_production_reviewer_feedback
    from .production_scene_metadata import install_production_scene_metadata
    from .production_settings import install_production_settings
    from .production_single_story_selection import install_production_single_story_selection
    from .production_source_deduplication import install_production_source_deduplication
    from .production_source_publishers import (
        install_production_source_publisher_canonicalization,
    )
    from .production_story_coherence import install_production_story_coherence
    from .production_video_qc import install_production_video_qc
    from .production_visual_quality import install_production_visual_quality
    from .production_visual_routing import install_production_visual_routing
    from .production_visual_semantics import install_production_visual_semantics
    from .production_voice_repair import install_production_voice_repair

    install_production_settings()
    install_production_audio_qc()
    install_production_source_publisher_canonicalization()
    install_production_narration_length_repair()
    install_production_content_gate()
    install_production_source_deduplication()
    install_production_relationship_grounding()
    install_production_story_coherence()
    install_production_single_story_selection()
    install_production_pacing()
    install_production_reviewer_feedback()
    install_production_voice_repair()
    install_production_visual_routing()
    install_production_visual_semantics()
    install_production_visual_quality()
    install_production_object_visuals()
    install_production_caption_zone()
    install_production_caption_quality()
    install_production_video_qc()
    install_production_renderer()
    install_production_scene_metadata()
    _INSTALLED = True
