"""Autonomous AI video growth worker with free-first Qwen generation and open-weight audio review."""

# Pure configuration authorities are installed at package import so production, tests, canaries,
# and library consumers resolve the same storyboard category and CLIP-safe prompt budget before
# any model or GPU resource is loaded.
from . import visual_storyboard_v30 as _visual_storyboard_v30
from .production_visual_clip_budget_v31 import install_production_visual_clip_budget_v31
from .production_visual_storyboard_priority_v30 import classify_claim_v30 as _classify_claim_v30

_visual_storyboard_v30.classify_claim = _classify_claim_v30
install_production_visual_clip_budget_v31()

__version__ = "1.3.1"
