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


def reward(obs: Observation) -> float:
    hours = max(obs.age_hours, 1)
    engagement = (obs.likes + 2 * obs.comments + 3 * obs.shares) / max(obs.views, 1)
    conversion = (obs.subscribers_gained - obs.subscribers_lost) / max(obs.views, 1)
    raw = 0.24 * math.log1p(obs.views / hours) / math.log(101) + 0.28 * min(engagement / 0.08, 1.5) + 0.30 * min(max(conversion, -0.02) / 0.01, 1.5) + 0.18 * min(obs.average_view_percentage / 100, 1.25)
    return max(0, min(raw, 1.5))


def all_strategies() -> list[Strategy]:
    return [Strategy(h, p, v, d, c) for h in HOOKS for p in PACING for v in VISUALS for d in DURATIONS for c in CTAS]


def select_strategy(observations: Iterable[Observation], seed: int) -> Strategy:
    observations = list(observations)
    strategies = all_strategies()
    rng = random.Random(seed)
    scores = {strategy.tag: [] for strategy in strategies}
    for obs in observations:
        if obs.strategy_tag in scores and obs.age_hours >= 24:
            scores[obs.strategy_tag].append(reward(obs))
    if sum(map(len, scores.values())) < 5:
        return rng.choice(strategies)
    best = None
    for strategy in strategies:
        values = scores[strategy.tag]
        successes = sum(min(max(value, 0), 1) for value in values)
        sample = rng.betavariate(1 + successes, 1 + len(values) - successes)
        if best is None or sample > best[0]:
            best = (sample, strategy)
    return best[1]
