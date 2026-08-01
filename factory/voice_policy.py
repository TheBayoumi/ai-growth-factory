from __future__ import annotations

from dataclasses import replace

from .models import VoiceContract
from .policy import Strategy


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def contract_for_strategy(base: VoiceContract, strategy: Strategy) -> VoiceContract:
    """Translate a growth-policy arm into an executable voice-performance contract."""
    target_wpm = base.target_wpm
    energy = base.energy
    warmth = base.warmth
    pitch_variation = base.pitch_variation
    hook_intensity = base.hook_intensity
    articulation = base.articulation
    style = base.baseline_style
    pause_style = base.pause_style

    if strategy.pacing == "fast":
        target_wpm += 10
        energy += 0.05
        pause_style = "tight rhetorical pauses; no dead air; preserve clarity around names and numbers"

    hook_adjustments = {
        "breaking": (0.08, 0.08, 0.02, "urgent but controlled"),
        "contrarian": (0.04, 0.07, -0.03, "analytical and confidently skeptical"),
        "practical": (0.02, 0.02, 0.05, "authoritative, useful, and approachable"),
        "myth-bust": (0.05, 0.06, 0.01, "precise, corrective, and non-combative"),
    }
    energy_delta, hook_delta, warmth_delta, style = hook_adjustments[strategy.hook]
    energy += energy_delta
    hook_intensity += hook_delta
    warmth += warmth_delta

    if strategy.hook in {"contrarian", "myth-bust"}:
        articulation += 0.04
    if strategy.hook == "breaking":
        pitch_variation += 0.04

    contract = replace(
        base,
        baseline_style=style,
        target_wpm=max(110, min(target_wpm, 195)),
        energy=_clamp(energy),
        warmth=_clamp(warmth),
        pitch_variation=_clamp(pitch_variation),
        hook_intensity=_clamp(hook_intensity),
        articulation=_clamp(articulation),
        pause_style=pause_style,
    )
    contract.validate()
    return contract
