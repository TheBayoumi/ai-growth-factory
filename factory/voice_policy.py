from __future__ import annotations

from dataclasses import replace

from .models import VoiceContract
from .policy import Strategy


def contract_for_strategy(base: VoiceContract, strategy: Strategy) -> VoiceContract:
    pace = base.target_wpm + (10 if strategy.pacing == "fast" else 0)
    styles = {
        "breaking": "urgent but controlled",
        "contrarian": "analytical and confidently skeptical",
        "practical": "authoritative, useful, and approachable",
        "myth-bust": "precise, corrective, and non-combative",
    }
    contract = replace(base, baseline_style=styles[strategy.hook], target_wpm=min(pace, 195), energy=min(base.energy + 0.05, 1), hook_intensity=min(base.hook_intensity + 0.06, 1))
    contract.validate()
    return contract
