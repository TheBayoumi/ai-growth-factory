from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production voice, editorial, and visual policies in deterministic order.

    Source attribution remains owned by factory.source_attributed_llm at the validated package
    boundary. The deleted source-index repair heuristic is intentionally not reintroduced.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_audio_qc import install_production_audio_qc
    from .production_caption_quality import install_production_caption_quality
    from .production_caption_layout_v32 import install_production_caption_layout_v32
    from .production_caption_zone import install_production_caption_zone
    from .production_content import install_production_content_gate
    from .production_editorial_v28 import install_production_editorial_v28
    from .production_editorial_wan_allocator_v38 import (
        install_production_editorial_wan_allocator_v38,
    )
    from .production_human_review_handoff_v61 import (
        install_production_human_review_handoff_v61,
    )
    from .production_narration_integrity_v28 import install_production_narration_integrity_v28
    from .production_narration_length import install_production_narration_length_repair
    from .production_object_visuals import install_production_object_visuals
    from .production_package_boundary_v54 import install_production_package_boundary_v54
    from .production_package_capacity_v46 import install_production_package_capacity_v46
    from .production_pacing import install_production_pacing
    from .production_qwen_omni_bitsandbytes_v28 import (
        install_production_qwen_omni_bitsandbytes_v28,
    )
    from .production_relationship_grounding import install_production_relationship_grounding
    from .production_renderer import install_production_renderer
    from .production_remotion_motion_qc_v53 import (
        install_production_remotion_motion_qc_v53,
    )
    from .production_reviewer_feedback import install_production_reviewer_feedback
    from .production_scene_metadata import install_production_scene_metadata
    from .production_settings import install_production_settings
    from .production_single_story_selection import install_production_single_story_selection
    from .production_source_deduplication import install_production_source_deduplication
    from .production_source_publishers import (
        install_production_source_publisher_canonicalization,
    )
    from .production_spoken_duration_capacity_v51 import (
        install_production_spoken_duration_capacity_v51,
    )
    from .production_story_coherence import install_production_story_coherence
    from .production_video_qc import install_production_video_qc
    from .production_visual_atomic_storyboard_v34 import (
        install_production_visual_atomic_storyboard_v34,
    )
    from .production_visual_convergence_v29 import (
        install_production_visual_convergence_v29,
    )
    from .production_visual_prompt_cleanup_v29 import (
        install_production_visual_prompt_cleanup_v29,
    )
    from .production_visual_quality import install_production_visual_quality
    from .production_visual_retry_grounding_v35 import (
        install_production_visual_retry_grounding_v35,
    )
    from .production_visual_review_json_v49 import (
        install_production_visual_review_json_v49,
    )
    from .production_visual_review_transport_v60 import (
        install_production_visual_review_transport_v60,
    )
    from .production_visual_reviewer_resilience_v59 import (
        install_production_visual_reviewer_resilience_v59,
    )
    from .production_visual_routing import install_production_visual_routing
    from .production_visual_runtime_v28 import install_production_visual_runtime_v28
    from .production_visual_semantic_review_v28 import (
        install_production_visual_semantic_review_v28,
    )
    from .production_visual_semantics import install_production_visual_semantics
    from .production_visual_storyboard_v30 import install_production_visual_storyboard_v30
    from .production_visual_subject_authority_v31 import (
        install_production_visual_subject_authority_v31,
    )
    from .production_vimax_editorial_grammar_v58 import (
        install_production_vimax_editorial_grammar_v58,
    )
    from .production_vimax_temporal_video_v55 import (
        install_production_vimax_temporal_video_v55,
    )
    from .production_vimax_visual_authority_v52 import (
        install_production_vimax_visual_authority_v52,
    )
    from .production_voice_bounds_v28 import install_production_voice_bounds_v28
    from .production_voice_calibration_v28 import install_production_voice_calibration_v28
    from .production_voice_capacity_v29 import install_production_voice_capacity_v29
    from .production_voice_clause_fallback_v33 import (
        install_production_voice_clause_fallback_v33,
    )
    from .production_voice_convergence_v28 import install_production_voice_convergence_v28
    from .production_voice_editorial_pacing_v28 import (
        install_production_voice_editorial_pacing_v28,
    )
    from .production_voice_micro_clause_fallback_v45 import (
        install_production_voice_micro_clause_fallback_v45,
    )
    from .production_voice_orphan_recovery_v50 import (
        install_production_voice_orphan_recovery_v50,
    )
    from .production_voice_repair import install_production_voice_repair
    from .production_voice_runtime_v28 import install_production_voice_runtime_v28
    from .production_voice_technical_identifier_v42 import (
        install_production_voice_technical_identifier_v42,
    )
    from .production_voice_technical_pacing_v43 import (
        install_production_voice_technical_pacing_v43,
    )

    install_production_settings()
    install_production_audio_qc()
    install_production_source_publisher_canonicalization()
    install_production_narration_length_repair()
    install_production_content_gate()
    # v46 converges generated package capacity before the spoken-duration validator.
    install_production_package_capacity_v46()
    install_production_spoken_duration_capacity_v51()
    # v54 is deliberately outside the earlier LLM wrappers. It removes only harmless scene
    # heading/body overflow before strict validation, preventing expensive trend churn for one
    # extra adjective while preserving every source, source_index, and hard narration gate.
    install_production_package_boundary_v54()
    install_production_narration_integrity_v28()
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

    # Voice quality remains fail-closed. v51 constrains script capacity before TTS; v50 repairs
    # generator-sized segment topology only. Neither changes the 138-146 WPM publication window
    # nor the 1.15x tempo ceiling. v42/v43 then apply auditable spoken-equivalent pacing and v45
    # remains the bounded last-resort TTS fallback.
    install_production_editorial_v28()
    install_production_editorial_wan_allocator_v38()
    install_production_voice_bounds_v28()
    install_production_voice_calibration_v28()
    install_production_voice_runtime_v28()
    install_production_qwen_omni_bitsandbytes_v28()
    install_production_voice_convergence_v28()
    install_production_voice_editorial_pacing_v28()
    install_production_voice_capacity_v29()
    install_production_voice_orphan_recovery_v50()
    install_production_voice_clause_fallback_v33()
    install_production_voice_technical_identifier_v42()
    install_production_voice_technical_pacing_v43()
    install_production_voice_micro_clause_fallback_v45()

    # Legacy visual adapters install first for compatibility. v41 is installed through v35 as the
    # final narration-grounded compiler, reviewer target, and failed-scene retry authority.
    install_production_visual_runtime_v28()
    install_production_visual_semantic_review_v28()
    # v49 keeps strict JSON parsing. v59 is a compatibility wrapper, while v60 owns the final
    # transport boundary: malformed/truncated model serialization becomes an explicit zero-score
    # scene retry and can never approve a frame or abort the complete production simulation.
    install_production_visual_review_json_v49()
    install_production_visual_reviewer_resilience_v59()
    install_production_visual_review_transport_v60()
    install_production_visual_convergence_v29()
    install_production_visual_prompt_cleanup_v29()
    install_production_visual_storyboard_v30()
    install_production_visual_subject_authority_v31()
    install_production_visual_atomic_storyboard_v34()
    install_production_visual_retry_grounding_v35()
    # Install the renderer last so the production path—not just unit tests—uses pixel-fitted ASS
    # captions and verifies actual libass output bounds before composition.
    install_production_caption_layout_v32()

    from . import production_editorial_v28
    from .production_editorial_compositor_v28 import compose_editorial_video_v28

    production_editorial_v28._compose_editorial_video = compose_editorial_video_v28

    # ViMax replaces planning only when explicitly enabled. Remotion is installed last so no
    # compatibility adapter can overwrite the selected final compositor.
    from .production_vimax_planning_v45 import install_production_vimax_planning_v45
    from .production_remotion_renderer_v45 import install_production_remotion_renderer_v45
    from .production_visual_audit_v45 import install_production_visual_audit_v45

    install_production_vimax_planning_v45()
    install_production_remotion_renderer_v45()
    install_production_visual_audit_v45()

    # v52/v53 repair ViMax semantic/motion authority and the legacy Remotion motion metric.
    install_production_vimax_visual_authority_v52()
    install_production_remotion_motion_qc_v53()
    # v55 is the ViMax temporal release authority. Every planned shot becomes native temporal
    # source media and image/Ken-Burns fallback is a hard failure.
    install_production_vimax_temporal_video_v55()
    # v58 installs after the temporal authority so it can replace the generic AI-lab physicalizer
    # with topic-aware, filmable editorial B-roll while retaining all 20 temporal shot contracts.
    install_production_vimax_editorial_grammar_v58()
    # v61 does not simulate approval. It prepares the complete machine-verified artifact for a
    # senior-editor HITL pass and records release_decision=blocked_pending_human_review until the
    # actual MP4/audio have been inspected.
    install_production_human_review_handoff_v61()
    _INSTALLED = True
