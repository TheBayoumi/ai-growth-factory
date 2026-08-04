from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production policies in deterministic order; v30 storyboard authority is final.

    Source attribution remains owned by factory.source_attributed_llm at the validated package
    boundary. The deleted source-index repair heuristic is intentionally not reintroduced.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_audio_qc import install_production_audio_qc
    from .production_caption_quality import install_production_caption_quality
    from .production_caption_zone import install_production_caption_zone
    from .production_content import install_production_content_gate
    from .production_editorial_v28 import install_production_editorial_v28
    from .production_narration_integrity_v28 import install_production_narration_integrity_v28
    from .production_narration_length import install_production_narration_length_repair
    from .production_object_visuals import install_production_object_visuals
    from .production_pacing import install_production_pacing
    from .production_qwen_omni_bitsandbytes_v28 import (
        install_production_qwen_omni_bitsandbytes_v28,
    )
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
    from .production_visual_convergence_v29 import (
        install_production_visual_convergence_v29,
    )
    from .production_visual_prompt_cleanup_v29 import (
        install_production_visual_prompt_cleanup_v29,
    )
    from .production_visual_quality import install_production_visual_quality
    from .production_visual_routing import install_production_visual_routing
    from .production_visual_runtime_v28 import install_production_visual_runtime_v28
    from .production_visual_semantic_review_v28 import (
        install_production_visual_semantic_review_v28,
    )
    from .production_visual_semantics import install_production_visual_semantics
    from .production_visual_storyboard_v30 import install_production_visual_storyboard_v30
    from .production_voice_bounds_v28 import install_production_voice_bounds_v28
    from .production_voice_calibration_v28 import install_production_voice_calibration_v28
    from .production_voice_capacity_v29 import install_production_voice_capacity_v29
    from .production_voice_convergence_v28 import install_production_voice_convergence_v28
    from .production_voice_editorial_pacing_v28 import (
        install_production_voice_editorial_pacing_v28,
    )
    from .production_voice_repair import install_production_voice_repair
    from .production_voice_runtime_v28 import install_production_voice_runtime_v28

    install_production_settings()
    install_production_audio_qc()
    install_production_source_publisher_canonicalization()
    install_production_narration_length_repair()
    install_production_content_gate()
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

    # Voice quality remains fail-closed. v29 replaces only the arbitrary segment-count ceiling
    # with narration-size-derived capacity; pace, fidelity, tempo, and perceptual review remain.
    install_production_editorial_v28()
    install_production_voice_bounds_v28()
    install_production_voice_calibration_v28()
    install_production_voice_runtime_v28()
    install_production_qwen_omni_bitsandbytes_v28()
    install_production_voice_convergence_v28()
    install_production_voice_editorial_pacing_v28()
    install_production_voice_capacity_v29()

    # Legacy visual adapters install first for compatibility. The configurable v30 storyboard
    # registry is the final authority for generation prompts, retry targets, and review evidence.
    install_production_visual_runtime_v28()
    install_production_visual_semantic_review_v28()
    install_production_visual_convergence_v29()
    install_production_visual_prompt_cleanup_v29()
    install_production_visual_storyboard_v30()

    from . import image_generator, production_editorial_v28, visual_pipeline
    from .production_editorial_compositor_v28 import compose_editorial_video_v28

    # The final visual pipeline owns stable selective retries, approved-frame caching,
    # storyboard-driven cross-environment diversity, and set-level perceptual enforcement.
    image_generator.generate_keyframes = visual_pipeline.generate_keyframes

    # Bind the CFR-safe compositor proven against mixed image/Wan inputs. This adapter forbids
    # source looping, gives stills deterministic motion, and forces yuv420p after subtitles.
    production_editorial_v28._compose_editorial_video = compose_editorial_video_v28
    _INSTALLED = True
