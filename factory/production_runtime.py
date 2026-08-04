from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production policies in deterministic order; v28 is the final authority."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_audio_qc import install_production_audio_qc
    from .production_caption_quality import install_production_caption_quality
    from .production_caption_zone import install_production_caption_zone
    from .production_content import install_production_content_gate
    from .production_editorial_v28 import install_production_editorial_v28
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

    # v28 installs last so legacy role templates, destructive mattes, fast pacing,
    # six-scene composition, and static-image exemptions cannot override it.
    install_production_editorial_v28()

    from . import image_generator, production_editorial_v28, visual_pipeline
    from .production_editorial_compositor_v28 import compose_editorial_video_v28

    # The legacy visual-quality adapter wraps visual_pipeline.generate_keyframes. Bind the
    # image-generator entry point used by v28 to that reviewed/regenerating implementation so
    # every editorial shot is still reviewed by Qwen before composition.
    image_generator.generate_keyframes = visual_pipeline.generate_keyframes

    # Bind the CFR-safe compositor proven against mixed image/Wan inputs. This adapter forbids
    # source looping, gives stills deterministic motion, and forces yuv420p after subtitles.
    production_editorial_v28._compose_editorial_video = compose_editorial_video_v28
    _INSTALLED = True
