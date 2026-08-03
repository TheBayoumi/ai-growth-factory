from __future__ import annotations

import os
from dataclasses import replace
from typing import Any


_INSTALLED = False


def install_production_settings() -> None:
    """Allow the production-only single-authority evidence contract.

    The base Settings schema retains its conservative two-source default for local and
    legacy deployments. After the production runtime is explicitly installed, an environment
    value of MIN_PRIMARY_SOURCES=1 is parsed through the existing validator and then narrowed
    to one authoritative source. Invalid values, including zero, still use the original
    fail-closed path.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .config import Settings

    original_from_env = Settings.from_env.__func__

    def production_from_env(cls: type[Settings]) -> Settings:
        raw = os.getenv("MIN_PRIMARY_SOURCES")
        single_authority = raw is not None and raw.strip() == "1"
        if not single_authority:
            return original_from_env(cls)

        previous = os.environ.get("MIN_PRIMARY_SOURCES")
        os.environ["MIN_PRIMARY_SOURCES"] = "2"
        try:
            settings = original_from_env(cls)
        finally:
            if previous is None:
                os.environ.pop("MIN_PRIMARY_SOURCES", None)
            else:
                os.environ["MIN_PRIMARY_SOURCES"] = previous
        return replace(settings, min_primary_sources=1)

    Settings.from_env = classmethod(production_from_env)
    _INSTALLED = True
