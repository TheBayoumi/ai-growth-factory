"""Autonomous AI video growth worker with free-first Qwen generation and open-weight audio review."""

# The v30 storyboard classifier is pure configuration. Install its specificity ordering at
# package import so every consumer—production runtime, tests, canaries, and library use—resolves
# the same claim category before any model or GPU resource is loaded.
from . import visual_storyboard_v30 as _visual_storyboard_v30
from .production_visual_storyboard_priority_v30 import classify_claim_v30 as _classify_claim_v30

_visual_storyboard_v30.classify_claim = _classify_claim_v30

__version__ = "1.3.1"
