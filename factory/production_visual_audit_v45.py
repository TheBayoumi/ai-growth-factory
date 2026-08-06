from __future__ import annotations

import shutil
from pathlib import Path


_INSTALLED = False


def _copy_new_artifacts(workdir: Path, destination: Path) -> None:
    visual_root = workdir / "visual-assets"
    candidates = (
        visual_root / "vimax-plan.json",
        visual_root / "render" / "remotion-render-spec.json",
        visual_root / "render" / "remotion-staged-render-spec.json",
        visual_root / "render" / "remotion-render-manifest.json",
        visual_root / "render" / "remotion-render.log",
    )
    for source in candidates:
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def install_production_visual_audit_v45() -> None:
    """Persist ViMax and Remotion evidence before temporary workdirs are removed."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import canary, pipeline

    current_pipeline = pipeline._persist_visual_audit
    current_canary = canary._copy_visual_audit

    def persist_visual_audit_v45(settings, workdir: Path) -> Path:
        destination = current_pipeline(settings, workdir)
        _copy_new_artifacts(workdir, destination)
        return destination

    def copy_canary_visual_audit_v45(workdir: Path, destination: Path) -> None:
        current_canary(workdir, destination)
        _copy_new_artifacts(workdir, destination)

    pipeline._persist_visual_audit = persist_visual_audit_v45
    canary._copy_visual_audit = copy_canary_visual_audit_v45
    _INSTALLED = True
