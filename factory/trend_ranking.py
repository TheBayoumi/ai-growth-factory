from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .feeds import SourceItem
from .trend_sources import TrendSnapshot


_TOKEN_RE = re.compile(r"[a-z0-9]{2,40}")
_STOP = {
    "about",
    "after",
    "against",
    "agent",
    "agents",
    "artificial",
    "from",
    "into",
    "intelligence",
    "latest",
    "machine",
    "model",
    "models",
    "news",
    "release",
    "technology",
    "that",
    "their",
    "this",
    "today",
    "using",
    "with",
}


@dataclass(frozen=True)
class TrendMatch:
    primary_url: str
    trend_url: str
    trend_publisher: str
    overlap_terms: tuple[str, ...]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_url": self.primary_url,
            "trend_url": self.trend_url,
            "trend_publisher": self.trend_publisher,
            "overlap_terms": list(self.overlap_terms),
            "score": self.score,
        }


@dataclass(frozen=True)
class TrendAlignment:
    ranked_sources: tuple[SourceItem, ...]
    matches: tuple[TrendMatch, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ranked_primary_urls": [item.url for item in self.ranked_sources],
            "matches": [match.as_dict() for match in self.matches],
        }


def _tokens(*values: str) -> set[str]:
    tokens = {token.casefold() for token in _TOKEN_RE.findall(" ".join(values).casefold())}
    return {token for token in tokens if token not in _STOP and len(token) > 2}


def _recency_score(item: SourceItem, *, now: datetime) -> float:
    age_hours = max(0.0, (now - item.published_at).total_seconds() / 3600.0)
    return max(0.0, 1.0 - age_hours / 168.0)


def _match_score(primary: SourceItem, trend: SourceItem) -> tuple[float, tuple[str, ...]]:
    primary_tokens = _tokens(primary.title, primary.summary)
    trend_tokens = _tokens(trend.title, trend.summary)
    overlap = tuple(sorted(primary_tokens & trend_tokens))
    if not overlap:
        return 0.0, ()
    denominator = math.sqrt(max(1, len(primary_tokens)) * max(1, len(trend_tokens)))
    lexical = len(overlap) / denominator
    popularity = 1.0 + math.log1p(max(0.0, trend.trend_score)) / 6.0
    title_bonus = 1.35 if _tokens(primary.title) & _tokens(trend.title) else 1.0
    return lexical * popularity * title_bonus, overlap


def align_primary_sources_to_trends(
    primary_sources: Iterable[SourceItem],
    snapshot: TrendSnapshot,
    *,
    limit: int = 30,
    now: datetime | None = None,
) -> TrendAlignment:
    """Rank factual primary sources using non-citable current trend signals.

    Trend pages influence which official stories are considered first. They are never
    returned as evidence candidates, so generated source URLs remain restricted to the
    primary publishers already enforced by the package validator.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    now = now or datetime.now(timezone.utc)
    scored: list[tuple[float, datetime, SourceItem, TrendMatch | None]] = []
    for primary in primary_sources:
        if not primary.is_primary:
            continue
        best_score = 0.0
        best_match: TrendMatch | None = None
        for trend in snapshot.items:
            score, overlap = _match_score(primary, trend)
            if score <= best_score:
                continue
            best_score = score
            best_match = TrendMatch(
                primary_url=primary.url,
                trend_url=trend.url,
                trend_publisher=trend.publisher,
                overlap_terms=overlap,
                score=round(score, 6),
            )
        combined = best_score * 4.0 + _recency_score(primary, now=now)
        scored.append((combined, primary.published_at, primary, best_match))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    ranked = tuple(row[2] for row in scored[:limit])
    matches = tuple(row[3] for row in scored[:limit] if row[3] is not None)
    return TrendAlignment(ranked_sources=ranked, matches=matches)
