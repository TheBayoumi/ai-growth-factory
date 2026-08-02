from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import requests

from .feeds import SourceItem, _parse


PRODUCT_HUNT_FEED = "https://www.producthunt.com/feed"
HUGGING_FACE_MODELS_API = "https://huggingface.co/api/models"
GITHUB_TRENDING_URL = "https://github.com/trending?since=daily"
HACKER_NEWS_TOP_API = "https://hacker-news.firebaseio.com/v0/topstories.json"
HACKER_NEWS_ITEM_API = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

_AI_PATTERN = re.compile(
    r"\b(?:ai|artificial intelligence|machine learning|deep learning|llm|vlm|"
    r"multimodal|agentic|agent|agents|inference|transformer|diffusion|text[- ]to[- ]image|"
    r"image[- ]to[- ]video|text[- ]to[- ]video|speech recognition|text[- ]to[- ]speech|"
    r"computer vision|rag|embedding|reasoning model|foundation model)\b",
    flags=re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_ARTICLE_RE = re.compile(
    r"<article[^>]+class=\"[^\"]*Box-row[^\"]*\"[^>]*>(.*?)</article>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REPO_RE = re.compile(r"href=\"/([^/\"?#]+/[^/\"?#]+)\"", flags=re.IGNORECASE)
_DESCRIPTION_RE = re.compile(r"<p[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
_STARS_TODAY_RE = re.compile(r"([\d,]+)\s+stars?\s+today", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TrendSnapshot:
    items: tuple[SourceItem, ...]
    provider_status: tuple[tuple[str, str], ...]
    generated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "provider_status": dict(self.provider_status),
            "items": [
                {
                    "publisher": item.publisher,
                    "title": item.title,
                    "url": item.url,
                    "summary": item.summary,
                    "published_at": item.published_at.isoformat(),
                    "source_kind": item.source_kind,
                    "trend_score": item.trend_score,
                }
                for item in self.items
            ],
        }


def _clean_html(value: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", value)).split())


def _is_ai_signal(*values: str) -> bool:
    return bool(_AI_PATTERN.search(" ".join(values)))


def _parse_datetime(value: Any, *, fallback: datetime | None = None) -> datetime:
    fallback = fallback or datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _within_age(item: SourceItem, *, now: datetime, max_age_hours: int) -> bool:
    age_seconds = (now - item.published_at).total_seconds()
    return -900 <= age_seconds <= max_age_hours * 3600


def _trend_item(
    *,
    publisher: str,
    title: str,
    url: str,
    summary: str,
    published_at: datetime,
    trend_score: float,
) -> SourceItem:
    return SourceItem(
        publisher=publisher,
        title=title.strip()[:300],
        url=url.strip(),
        summary=summary.strip()[:1200],
        published_at=published_at,
        source_kind="trend",
        trend_score=round(max(0.0, float(trend_score)), 4),
    )


def parse_product_hunt_feed(content: bytes, *, limit: int = 12) -> list[SourceItem]:
    parsed = _parse(content, "Product Hunt")
    result: list[SourceItem] = []
    for index, item in enumerate(parsed):
        if not _is_ai_signal(item.title, item.summary):
            continue
        result.append(
            _trend_item(
                publisher="Product Hunt",
                title=item.title,
                url=item.url,
                summary=(
                    "Product Hunt launch signal. Treat this as popularity and recency context, "
                    f"not independent product verification. {item.summary}"
                ),
                published_at=item.published_at,
                trend_score=max(1.0, 100.0 - index * 5.0),
            )
        )
        if len(result) >= limit:
            break
    return result


def parse_hugging_face_models(
    payload: Any,
    *,
    now: datetime | None = None,
    limit: int = 12,
) -> list[SourceItem]:
    if not isinstance(payload, list):
        return []
    now = now or datetime.now(timezone.utc)
    items: list[SourceItem] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id") or raw.get("modelId") or "").strip()
        if not model_id:
            continue
        pipeline_tag = str(raw.get("pipeline_tag") or "unknown")
        tags = [str(tag) for tag in raw.get("tags", []) if isinstance(tag, str)][:8]
        likes = int(raw.get("likes") or 0)
        downloads = int(raw.get("downloads") or 0)
        raw_score = raw.get("trendingScore")
        try:
            trend_score = float(raw_score)
        except (TypeError, ValueError):
            trend_score = likes * 2.0 + min(downloads / 1000.0, 500.0) + max(0, 50 - index)
        updated = _parse_datetime(raw.get("lastModified"), fallback=now)
        items.append(
            _trend_item(
                publisher="Hugging Face Trending",
                title=f"{model_id} — {pipeline_tag}",
                url=f"https://huggingface.co/{model_id}",
                summary=(
                    "Hugging Face Hub trend signal. Treat popularity metrics as discovery context, "
                    f"not proof of model quality. task={pipeline_tag}; likes={likes}; "
                    f"downloads={downloads}; tags={', '.join(tags)}"
                ),
                published_at=updated,
                trend_score=trend_score,
            )
        )
        if len(items) >= limit:
            break
    return items


def parse_github_trending(
    content: str,
    *,
    now: datetime | None = None,
    limit: int = 12,
) -> list[SourceItem]:
    now = now or datetime.now(timezone.utc)
    items: list[SourceItem] = []
    for index, article in enumerate(_ARTICLE_RE.findall(content)):
        repo_match = _REPO_RE.search(article)
        if repo_match is None:
            continue
        repository = repo_match.group(1).strip()
        description_match = _DESCRIPTION_RE.search(article)
        description = _clean_html(description_match.group(1)) if description_match else ""
        if not _is_ai_signal(repository.replace("/", " "), description):
            continue
        stars_match = _STARS_TODAY_RE.search(_clean_html(article))
        stars_today = int(stars_match.group(1).replace(",", "")) if stars_match else 0
        items.append(
            _trend_item(
                publisher="GitHub Trending",
                title=repository,
                url=f"https://github.com/{repository}",
                summary=(
                    "GitHub daily trend signal. Treat star velocity as developer-interest context, "
                    f"not proof of production quality. stars_today={stars_today}; {description}"
                ),
                published_at=now,
                trend_score=float(stars_today or max(1, 30 - index)),
            )
        )
        if len(items) >= limit:
            break
    return items


def parse_hacker_news_items(
    payloads: Iterable[Any],
    *,
    now: datetime | None = None,
    max_age_hours: int = 48,
    limit: int = 12,
) -> list[SourceItem]:
    now = now or datetime.now(timezone.utc)
    items: list[SourceItem] = []
    for raw in payloads:
        if not isinstance(raw, dict) or raw.get("type") != "story":
            continue
        title = str(raw.get("title") or "").strip()
        original_url = str(raw.get("url") or "").strip()
        if not title or not _is_ai_signal(title, original_url):
            continue
        published = _parse_datetime(raw.get("time"), fallback=now)
        item_id = int(raw.get("id") or 0)
        if not item_id:
            continue
        score = int(raw.get("score") or 0)
        comments = int(raw.get("descendants") or 0)
        item = _trend_item(
            publisher="Hacker News",
            title=title,
            url=f"https://news.ycombinator.com/item?id={item_id}",
            summary=(
                "Hacker News discussion signal. Treat community engagement as discovery context, "
                f"not independent verification. points={score}; comments={comments}; "
                f"original_url={original_url or 'not supplied'}"
            ),
            published_at=published,
            trend_score=score + comments * 0.5,
        )
        if _within_age(item, now=now, max_age_hours=max_age_hours):
            items.append(item)
    return sorted(items, key=lambda item: (item.trend_score, item.published_at), reverse=True)[:limit]


def fetch_product_hunt_trends(*, timeout_seconds: float = 10.0) -> list[SourceItem]:
    response = requests.get(
        PRODUCT_HUNT_FEED,
        headers={"User-Agent": "AIGrowthFactory/1.4 trend-research"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_product_hunt_feed(response.content)


def fetch_hugging_face_trends(*, timeout_seconds: float = 10.0) -> list[SourceItem]:
    response = requests.get(
        HUGGING_FACE_MODELS_API,
        params={"sort": "trendingScore", "direction": "-1", "limit": 20},
        headers={"User-Agent": "AIGrowthFactory/1.4 trend-research"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_hugging_face_models(response.json())


def fetch_github_trends(*, timeout_seconds: float = 10.0) -> list[SourceItem]:
    response = requests.get(
        GITHUB_TRENDING_URL,
        headers={
            "User-Agent": "AIGrowthFactory/1.4 trend-research",
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_github_trending(response.text)


def fetch_hacker_news_trends(
    *,
    max_age_hours: int,
    timeout_seconds: float = 10.0,
) -> list[SourceItem]:
    response = requests.get(HACKER_NEWS_TOP_API, timeout=timeout_seconds)
    response.raise_for_status()
    story_ids = response.json()
    if not isinstance(story_ids, list):
        return []
    selected_ids = [int(value) for value in story_ids[:40] if isinstance(value, int)]

    def fetch_one(item_id: int) -> Any:
        item_response = requests.get(
            HACKER_NEWS_ITEM_API.format(item_id=item_id),
            timeout=min(timeout_seconds, 6.0),
        )
        item_response.raise_for_status()
        return item_response.json()

    payloads: list[Any] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, item_id): item_id for item_id in selected_ids}
        for future in as_completed(futures):
            try:
                payloads.append(future.result())
            except Exception:
                continue
    return parse_hacker_news_items(payloads, max_age_hours=max_age_hours)


TrendFetcher = Callable[..., list[SourceItem]]


def fetch_trend_snapshot(
    *,
    max_age_hours: int,
    timeout_seconds: float = 10.0,
    per_provider_limit: int = 6,
    total_limit: int = 24,
    fetchers: tuple[tuple[str, TrendFetcher], ...] | None = None,
) -> TrendSnapshot:
    """Collect independent discovery signals while failing soft per provider.

    Trend signals never replace primary evidence. Callers pass them to the research
    prompt as non-citable ranking context, while factual claims and source URLs remain
    restricted to official primary-source feed entries.
    """
    if not 1 <= max_age_hours <= 168:
        raise ValueError("max_age_hours must be between 1 and 168")
    now = datetime.now(timezone.utc)
    providers = fetchers or (
        ("product_hunt", fetch_product_hunt_trends),
        ("hugging_face", fetch_hugging_face_trends),
        ("github_trending", fetch_github_trends),
        ("hacker_news", fetch_hacker_news_trends),
    )
    statuses: list[tuple[str, str]] = []
    collected: list[SourceItem] = []
    for provider, fetcher in providers:
        try:
            if provider == "hacker_news":
                items = fetcher(
                    max_age_hours=max_age_hours,
                    timeout_seconds=timeout_seconds,
                )
            else:
                items = fetcher(timeout_seconds=timeout_seconds)
            fresh = [
                item
                for item in items
                if item.source_kind == "trend"
                and _within_age(item, now=now, max_age_hours=max_age_hours)
            ][:per_provider_limit]
            collected.extend(fresh)
            statuses.append((provider, f"ok:{len(fresh)}"))
        except Exception as exc:
            statuses.append((provider, f"error:{type(exc).__name__}"))

    unique: dict[str, SourceItem] = {}
    for item in collected:
        existing = unique.get(item.url)
        if existing is None or item.trend_score > existing.trend_score:
            unique[item.url] = item
    ranked = sorted(
        unique.values(),
        key=lambda item: (item.trend_score, item.published_at),
        reverse=True,
    )[:total_limit]
    return TrendSnapshot(tuple(ranked), tuple(statuses), now)
