from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from xml.etree import ElementTree

import requests


@dataclass(frozen=True)
class SourceItem:
    publisher: str
    title: str
    url: str
    summary: str
    published_at: datetime

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.publisher}|{self.title}|{self.url}".encode()).hexdigest()[:16]


FEEDS: tuple[tuple[str, str], ...] = (
    ("OpenAI", "https://openai.com/news/rss.xml"),
    ("Google AI", "https://blog.google/technology/ai/rss/"),
    ("Anthropic", "https://www.anthropic.com/rss.xml"),
    ("Microsoft Research", "https://www.microsoft.com/en-us/research/feed/"),
    ("NVIDIA", "https://blogs.nvidia.com/feed/"),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
)

_TAG_RE = re.compile(r"<[^>]+>")


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
        date_text = _child_text(node, {"published", "updated", "pubdate", "date"})
        if title and url:
            items.append(SourceItem(publisher, title, url, summary[:1200], _date(date_text)))
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
    return {item.publisher for item in items}
