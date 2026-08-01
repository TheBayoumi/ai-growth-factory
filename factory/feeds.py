from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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


FEEDS = (
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


def _child(node: ElementTree.Element, names: set[str]) -> str:
    for child in list(node):
        if _local(child.tag) in names:
            return "".join(child.itertext()).strip()
    return ""


def _link(node: ElementTree.Element) -> str:
    for child in list(node):
        if _local(child.tag) == "link":
            return (child.attrib.get("href") or child.text or "").strip()
    return ""


def _date(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def fetch_recent(max_age_hours: int, timeout_seconds: float = 10) -> list[SourceItem]:
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600
    items = []
    for publisher, url in FEEDS:
        try:
            response = requests.get(url, headers={"User-Agent": "AIGrowthFactory/1.3.1"}, timeout=timeout_seconds)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for node in [item for item in root.iter() if _local(item.tag) in {"item", "entry"}][:12]:
                title = _clean(_child(node, {"title"}))
                link = _link(node) or _child(node, {"guid", "id"})
                published = _date(_child(node, {"published", "updated", "pubdate", "date"}))
                if title and link and published.timestamp() >= cutoff:
                    items.append(SourceItem(publisher, title, link, _clean(_child(node, {"summary", "description", "content", "encoded"}))[:1200], published))
        except Exception:
            continue
    return sorted({item.fingerprint: item for item in items}.values(), key=lambda item: item.published_at, reverse=True)


def publishers(items: list[SourceItem]) -> set[str]:
    return {item.publisher for item in items}
