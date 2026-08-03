from __future__ import annotations

from typing import Any

from .feeds import SourceItem
from .local_llm import LocalLLMError


_INSTALLED = False


def canonicalize_source_publishers(
    raw: dict[str, Any],
    sources: list[SourceItem],
) -> dict[str, Any]:
    """Derive publisher metadata from validated evidence URLs.

    The model may select evidence URLs and write claims, but publisher names are catalog
    metadata and must never be copied probabilistically. Unknown or conflicting catalog
    URLs still fail closed in the normal package validator.
    """
    selected = raw.get("source_urls")
    if not isinstance(selected, list):
        return raw

    publisher_by_url: dict[str, str] = {}
    for source in sources:
        url = source.url.strip()
        publisher = source.publisher.strip()
        existing = publisher_by_url.get(url)
        if existing is not None and existing.casefold() != publisher.casefold():
            raise LocalLLMError(
                f"Supplied source catalog has conflicting publishers for URL: {url}"
            )
        publisher_by_url[url] = publisher

    corrected = dict(raw)
    corrected["source_publishers"] = [
        publisher_by_url.get(str(raw_url).strip(), "") for raw_url in selected
    ]
    return corrected


def install_production_source_publisher_canonicalization() -> None:
    """Install deterministic publisher attribution before package validation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original = local_llm._package_from_raw

    def package_with_canonical_publishers(
        settings: Any,
        sources: list[SourceItem],
        raw: dict[str, Any],
    ) -> Any:
        return original(settings, sources, canonicalize_source_publishers(raw, sources))

    local_llm._package_from_raw = package_with_canonical_publishers
    _INSTALLED = True
