from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production-only content, pacing, voice-repair, and render policies.

    Scene attribution is handled by factory.source_attributed_llm at the package
    validation boundary. No heuristic source-index hook is installed here.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_content import install_production_content_gate
    from .production_pacing import install_production_pacing
    from .production_renderer import install_production_renderer
    from .production_voice_repair import install_production_voice_repair

    install_production_content_gate()
    install_production_pacing()
    install_production_voice_repair()
    install_production_renderer()
    _INSTALLED = True
