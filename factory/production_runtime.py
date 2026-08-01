from __future__ import annotations


_INSTALLED = False


def install_production_runtime() -> None:
    """Install production-only editorial, pacing, source, and render policies."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_content import install_production_content_gate
    from .production_pacing import install_production_pacing
    from .production_renderer import install_production_renderer
    from .source_index_repair import install_source_index_repair

    install_source_index_repair()
    install_production_content_gate()
    install_production_pacing()
    install_production_renderer()
    _INSTALLED = True
