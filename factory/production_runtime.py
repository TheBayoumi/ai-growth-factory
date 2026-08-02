from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production-only content, pacing, voice, visual, and render policies.

    Scene attribution is handled by factory.source_attributed_llm at the package
    validation boundary. Visual model routing is deterministic after the director has
    supplied semantic roles and prompts, so an exact resource count cannot depend on
    free-form model compliance.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_content import install_production_content_gate
    from .production_pacing import install_production_pacing
    from .production_renderer import install_production_renderer
    from .production_visual_routing import install_production_visual_routing
    from .production_voice_repair import install_production_voice_repair

    install_production_content_gate()
    install_production_pacing()
    install_production_voice_repair()
    install_production_visual_routing()
    install_production_renderer()
    _INSTALLED = True
