from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable


HOOKS = ("practical", "contrarian", "breaking", "myth-bust")
PACING = ("fast", "balanced")
VISUALS = ("kinetic", "dashboard", "cinematic")
DURATIONS = ("55-62", "63-72", "73-82")
CTAS = ("subscribe", "newsletter", "product", "consulting")


@dataclass(frozen=True)
class Strategy:
    hook: str
    pacing: str
    visual: str
    duration: str
    cta: str

    @property
    def key(self) -> str:
        return ":".join((self.hook, self.pacing, self.visual, self.duration, self.cta))

    @property
    def tag(self) -> str:
        return "agfs-" + hashlib.sha1(self.key.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Observation:
    strategy_tag: str
    views: int
    likes: int
    comments: int
    shares: int
    subscribers_gained: int
    subscribers_lost: int
    average_view_percentage: float
    age_hours: float


def all_strategies() -> list[Strategy]:
    return [
        Strategy(h, p, v, d, c)
        for h in HOOKS
        for p in PACING
        for v in VISUALS
        for d in DURATIONS
        for c in CTAS
    ]


def reward(obs: Observation) -> float:
    hours = max(obs.age_hours, 1.0)
    vph = obs.views / hours
    engagement = (obs.likes + 2 * obs.comments + 3 * obs.shares) / max(obs.views, 1)
    subscriber_conversion = (obs.subscribers_gained - obs.subscribers_lost) / max(obs.views, 1)
    retention = max(0.0, min(obs.average_view_percentage / 100.0, 1.25))
    raw = (
        0.24 * math.log1p(vph) / math.log(101)
        + 0.28 * min(engagement / 0.08, 1.5)
        + 0.30 * min(max(subscriber_conversion, -0.02) / 0.01, 1.5)
        + 0.18 * retention
    )
    return max(0.0, min(raw, 1.5))


def select_strategy(observations: Iterable[Observation], seed: int) -> Strategy:
    observations = list(observations)
    strategies = all_strategies()
    rng = random.Random(seed)
    tag_map = {strategy.tag: strategy for strategy in strategies}
    scores: dict[str, list[float]] = {strategy.tag: [] for strategy in strategies}
    for obs in observations:
        if obs.strategy_tag in scores and obs.age_hours >= 24:
            scores[obs.strategy_tag].append(reward(obs))

    mature = sum(len(values) for values in scores.values())
    recent = [reward(obs) for obs in observations if obs.age_hours >= 24][-5:]
    baseline = sorted(recent)[len(recent) // 2] if recent else 0.0
    recovery = len(recent) >= 3 and recent[-1] < 0.8 * max(baseline, 0.2)

    if mature < 5:
        return rng.choice(strategies)

    best: tuple[float, Strategy] | None = None
    for tag, strategy in tag_map.items():
        values = scores[tag]
        successes = sum(min(max(value, 0.0), 1.0) for value in values)
        failures = len(values) - successes
        sample = rng.betavariate(1.0 + successes, 1.0 + failures)
        if recovery and not values:
            sample *= 0.25
        if best is None or sample > best[0]:
            best = (sample, strategy)
    assert best is not None
    return best[1]
