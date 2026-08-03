from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production-only content, source, voice, visual, render, and QC policies.

    Voice normalization uses strict peak headroom before any voice module binds the audio
    helpers, and deterministic QC evidence overrides stale reviewer feedback. Publisher
    names are canonicalized from selected evidence URLs before package validation. Scene
    attribution is then handled by factory.source_attributed_llm at the exact package
    boundary. Near-complete narration length is stabilized before the editorial wrapper.
    Unsupported publisher relationships are grounded in explicit independent-source
    language before the same validator runs. Incomplete but otherwise valid Qwen Omni
    feedback is normalized before the bounded selective repair loop executes. Executable
    visual prompts must stay semantically distinct, Wan motion is bound to each scene's
    actual subject, and temporal QC excludes the separate caption layer while retaining
    strict checks for generated video scenes.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_audio_qc import install_production_audio_qc
    from .production_content import install_production_content_gate
    from .production_narration_length import install_production_narration_length_repair
    from .production_pacing import install_production_pacing
    from .production_relationship_grounding import install_production_relationship_grounding
    from .production_renderer import install_production_renderer
    from .production_reviewer_feedback import install_production_reviewer_feedback
    from .production_source_publishers import (
        install_production_source_publisher_canonicalization,
    )
    from .production_video_qc import install_production_video_qc
    from .production_visual_routing import install_production_visual_routing
    from .production_visual_semantics import install_production_visual_semantics
    from .production_voice_repair import install_production_voice_repair

    install_production_audio_qc()
    install_production_source_publisher_canonicalization()
    install_production_narration_length_repair()
    install_production_content_gate()
    install_production_relationship_grounding()
    install_production_pacing()
    install_production_reviewer_feedback()
    install_production_voice_repair()
    install_production_visual_routing()
    install_production_visual_semantics()
    install_production_video_qc()
    install_production_renderer()
    _INSTALLED = True
