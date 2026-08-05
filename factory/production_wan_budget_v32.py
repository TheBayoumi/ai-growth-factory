from __future__ import annotations

from typing import Sequence

from .editorial_timeline import ShotSpec
from .video_profile import VideoProfile


_INSTALLED = False


def select_wan_indices_v32(
    shots: Sequence[ShotSpec],
    profile: VideoProfile,
) -> set[int]:
    """Select the configured Wan budget deterministically across the timeline.

    Shot zero is always the opening hero. Remaining Wan shots are chosen from assets that fit the
    Wan duration ceiling and are distributed around evenly spaced timeline quantiles. The selector
    supports every validated ``wan_shots`` value instead of assuming the legacy three-shot budget.
    """
    ordered = sorted(shots, key=lambda shot: shot.shot_id)
    if not ordered:
        raise ValueError("Cannot select Wan shots from an empty editorial timeline")
    if [shot.shot_id for shot in ordered] != list(range(len(ordered))):
        raise ValueError("Wan selection requires contiguous editorial shot IDs")
    if profile.wan_shots > len(ordered):
        raise ValueError(
            f"Configured Wan budget {profile.wan_shots} exceeds {len(ordered)} editorial shots"
        )
    if ordered[0].duration_seconds > profile.maximum_wan_shot_seconds + 1e-6:
        raise ValueError("The opening Wan shot exceeds its configured duration")

    selected = {ordered[0].shot_id}
    remaining_budget = profile.wan_shots - 1
    if remaining_budget == 0:
        return selected

    eligible = [
        shot
        for shot in ordered[1:]
        if shot.duration_seconds <= profile.maximum_wan_shot_seconds + 1e-6
    ]
    if len(eligible) < remaining_budget:
        raise ValueError(
            "Not enough short shots are available for the configured Wan budget: "
            f"need {remaining_budget}, found {len(eligible)}"
        )

    total_duration = max(
        shot.start_seconds + shot.duration_seconds
        for shot in ordered
    )
    targets = [
        total_duration * ordinal / (remaining_budget + 1)
        for ordinal in range(1, remaining_budget + 1)
    ]

    for target in targets:
        candidates = [shot for shot in eligible if shot.shot_id not in selected]
        if not candidates:
            raise ValueError("Could not select the exact Wan shot budget")
        candidate = min(
            candidates,
            key=lambda shot: (
                abs((shot.start_seconds + shot.duration_seconds / 2.0) - target),
                shot.shot_id,
            ),
        )
        selected.add(candidate.shot_id)

    if len(selected) != profile.wan_shots:
        raise ValueError(
            f"Wan selector produced {len(selected)} shots for budget {profile.wan_shots}"
        )
    return selected


def install_production_wan_budget_v32() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import editorial_timeline

    editorial_timeline._wan_indices = select_wan_indices_v32
    _INSTALLED = True
