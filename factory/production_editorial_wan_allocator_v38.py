from __future__ import annotations

from . import editorial_timeline
from .editorial_timeline import StoryBeat
from .video_profile import VideoProfile


_INSTALLED = False
_BASE_ALLOCATOR = editorial_timeline._allocate_durations
_WAN_BOUNDS_ERROR = "Opening Wan shot cannot fit inside the configured duration bounds"


def allocate_editorial_durations_v38(
    beats: list[StoryBeat],
    counts: list[int],
    profile: VideoProfile,
) -> list[list[float]]:
    """Use spare shot capacity when the opening Wan ceiling cannot be redistributed.

    The legacy allocator first tries to move excess time from the opening Wan shot into later
    shots from the same spoken beat. A balanced two-shot opening can leave less recipient room
    than the Wan correction requires even though a valid three-shot allocation exists. Split only
    that spoken beat, preserve all timing bounds, and retry through the original allocator.
    """
    while True:
        try:
            return _BASE_ALLOCATOR(beats, counts, profile)
        except ValueError as exc:
            if str(exc) != _WAN_BOUNDS_ERROR or not beats or not counts:
                raise

            next_count = counts[0] + 1
            can_split = (
                sum(counts) < profile.maximum_shots
                and beats[0].duration_seconds / next_count
                >= profile.minimum_shot_seconds - 1e-9
            )
            if not can_split:
                raise

            counts[0] = next_count
            print(
                "[editorial-wan-v38] split opening beat to "
                f"{next_count} shots after bounded redistribution exhausted",
                flush=True,
            )


def install_production_editorial_wan_allocator_v38() -> None:
    """Install profile-driven Wan selection and bounded opening-Wan recovery."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .production_wan_budget_v32 import install_production_wan_budget_v32

    install_production_wan_budget_v32()
    editorial_timeline._allocate_durations = allocate_editorial_durations_v38
    _INSTALLED = True
