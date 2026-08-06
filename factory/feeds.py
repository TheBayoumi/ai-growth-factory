from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests


@dataclass(frozen=True)
class SourceItem:
    publisher: str
    title: str
    url: str
    summary: str
    published_at: datetime
    source_kind: str = "primary"
    trend_score: float = 0.0
    author: str = ""

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.publisher}|{self.title}|{self.url}".encode()).hexdigest()[:16]

    @property
    def is_primary(self) -> bool:
        return self.source_kind == "primary"

    @property
    def authority(self) -> str:
        """Return the organization/person responsible for the article, not its host."""
        return source_authority(self)


@dataclass(frozen=True)
class SourceSelection:
    items: tuple[SourceItem, ...]
    publishers: frozenset[str]
    max_age_hours: int

    @property
    def publisher_count(self) -> int:
        return len(self.publishers)


FetchRecent = Callable[..., list[SourceItem]]


FEEDS: tuple[tuple[str, str], ...] = (
    ("OpenAI", "https://openai.com/news/rss.xml"),
    ("Google AI", "https://blog.google/technology/ai/rss/"),
    ("Anthropic", "https://www.anthropic.com/rss.xml"),
    ("Microsoft Research", "https://www.microsoft.com/en-us/research/feed/"),
    ("NVIDIA", "https://blogs.nvidia.com/feed/"),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
)

_TAG_RE = re.compile(r"<[^>]+>")


def _humanize_namespace(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", value).strip()
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cleaned)
    return " ".join(cleaned.split())


def source_authority(source: SourceItem) -> str:
    """Resolve authorial authority while preserving publisher as distribution metadata.

    Hugging Face team articles encode the author/organization in ``/blog/<namespace>/``.
    Treating the hosting feed as the announcing company caused the rejected LFM2.5 render.
    """
    parsed = urlparse(source.url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.hostname and parsed.hostname.casefold().endswith("huggingface.co"):
        if len(parts) >= 3 and parts[0].casefold() == "blog":
            namespace = _humanize_namespace(parts[1])
            if namespace and namespace.casefold() not in {"hugging face", "huggingface"}:
                return namespace
    author = " ".join(source.author.split()).strip()
    return author or source.publisher.strip()


def _clean(value: str | None) -> str:
    return html.unescape(_TAG_RE.sub(" ", value or "")).replace("\n", " ").strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node: ElementTree.Element, names: set[str]) -> str:
    for child in list(node):
        if _local(child.tag) in names:
            return "".join(child.itertext()).strip()
    return ""


def _link(node: ElementTree.Element) -> str:
    for child in list(node):
        if _local(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        rel = child.attrib.get("rel", "alternate")
        if href and rel in {"alternate", ""}:
            return href.strip()
        if child.text and child.text.strip():
            return child.text.strip()
    return ""


def _date(value: str) -> datetime:
    value = value.strip()
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse(content: bytes, publisher: str) -> list[SourceItem]:
    root = ElementTree.fromstring(content)
    nodes = [node for node in root.iter() if _local(node.tag) in {"item", "entry"}]
    items: list[SourceItem] = []
    for node in nodes[:12]:
        title = _clean(_child_text(node, {"title"}))
        url = _link(node) or _child_text(node, {"guid", "id"})
        summary = _clean(_child_text(node, {"summary", "description", "content", "encoded"}))
        author = _clean(_child_text(node, {"author", "creator"}))
        date_text = _child_text(node, {"published", "updated", "pubdate", "date"})
        if title and url:
            items.append(
                SourceItem(
                    publisher,
                    title,
                    url,
                    summary[:1200],
                    _date(date_text),
                    author=author[:240],
                )
            )
    return items


def fetch_recent(*, max_age_hours: int, timeout_seconds: float = 10.0) -> list[SourceItem]:
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max_age_hours * 3600
    items: list[SourceItem] = []
    headers = {"User-Agent": "AIGrowthFactory/1.0 (+https://vercel.app)"}
    for publisher, url in FEEDS:
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            parsed = _parse(response.content, publisher)
        except Exception:
            continue
        for item in parsed:
            timestamp = item.published_at.timestamp()
            if cutoff <= timestamp <= now.timestamp() + 900:
                items.append(item)
    unique: dict[str, SourceItem] = {item.fingerprint: item for item in items}
    return sorted(unique.values(), key=lambda item: item.published_at, reverse=True)


def publishers(items: Iterable[SourceItem]) -> set[str]:
    return {item.publisher for item in items if item.is_primary}


def fetch_diverse_recent(
    *,
    max_age_hours: int,
    min_publishers: int,
    fallback_max_age_hours: int = 168,
    timeout_seconds: float = 10.0,
    fetcher: FetchRecent = fetch_recent,
) -> SourceSelection:
    """Select fresh primary sources, expanding once when diversity is insufficient.

    The fallback is bounded to seven days and never weakens the independent-publisher
    requirement. Discovery-only trend signals are deliberately excluded from publisher
    diversity and are collected separately by ``factory.trend_sources``.
    """
    if not 1 <= max_age_hours <= 168:
        raise ValueError("max_age_hours must be between 1 and 168")
    if min_publishers < 1:
        raise ValueError("min_publishers must be at least 1")

    fallback_window = min(max(max_age_hours, fallback_max_age_hours), 168)
    windows = (max_age_hours,) if fallback_window == max_age_hours else (max_age_hours, fallback_window)
    best: SourceSelection | None = None

    for window in windows:
        items = tuple(
            item
            for item in fetcher(max_age_hours=window, timeout_seconds=timeout_seconds)
            if item.is_primary
        )
        selection = SourceSelection(
            items=items,
            publishers=frozenset(publishers(items)),
            max_age_hours=window,
        )
        if best is None or (selection.publisher_count, len(selection.items)) > (
            best.publisher_count,
            len(best.items),
        ):
            best = selection
        if selection.publisher_count >= min_publishers:
            return selection

    if best is None:
        raise RuntimeError("Source selection did not execute")
    return best
