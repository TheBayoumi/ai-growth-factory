from __future__ import annotations

import os
import re
from typing import Any


_INSTALLED = False
_MINIMUM_REALIZED_TRANSITIONS = 2
_PACKAGE_SCENE_RE = re.compile(r"(?:^|;)\s*package_scene:(\d+)\b", re.IGNORECASE)


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _package_scene_index(shot: dict[str, Any]) -> int | None:
    match = _PACKAGE_SCENE_RE.search(str(shot.get("purpose") or ""))
    return int(match.group(1)) if match else None


def _transition_frames(
    outgoing: dict[str, Any],
    incoming: dict[str, Any],
    requested_frames: int,
) -> int:
    if requested_frames <= 0:
        return 0
    outgoing_renderer = str(outgoing.get("renderer") or "")
    incoming_renderer = str(incoming.get("renderer") or "")
    should_dissolve = outgoing_renderer == "image_motion"
    if outgoing_renderer == "video_clip" and incoming_renderer == "video_clip":
        outgoing_beat = _package_scene_index(outgoing)
        incoming_beat = _package_scene_index(incoming)
        should_dissolve = (
            outgoing_beat is not None
            and incoming_beat is not None
            and outgoing_beat != incoming_beat
        )
    if not should_dissolve:
        return 0
    return max(
        0,
        min(
            requested_frames,
            int(outgoing.get("duration_in_frames") or 0) // 3,
            int(incoming.get("duration_in_frames") or 0) // 3,
        ),
    )


def build_remotion_editorial_transition_evidence_v57(
    spec_payload: dict[str, Any],
) -> dict[str, Any]:
    """Audit deliberate beat-boundary dissolves and continuity-preserving hard cuts."""
    shots = list(spec_payload.get("shots") or [])
    requested_frames = int(spec_payload.get("transition_frames") or 0)
    transitions: list[dict[str, Any]] = []
    hard_cuts: list[dict[str, Any]] = []

    for index in range(max(0, len(shots) - 1)):
        outgoing = dict(shots[index])
        incoming = dict(shots[index + 1])
        outgoing_id = int(outgoing.get("shot_id", index))
        incoming_id = int(incoming.get("shot_id", index + 1))
        if incoming_id != outgoing_id + 1:
            raise ValueError(
                f"Remotion transition boundary is not contiguous: {outgoing_id}->{incoming_id}"
            )
        frames = _transition_frames(outgoing, incoming, requested_frames)
        boundary = {
            "outgoing_shot_id": outgoing_id,
            "incoming_shot_id": incoming_id,
            "start_frame": int(incoming.get("start_frame") or 0),
            "outgoing_package_scene_index": _package_scene_index(outgoing),
            "incoming_package_scene_index": _package_scene_index(incoming),
        }
        if frames > 0:
            transitions.append(
                {
                    **boundary,
                    "duration_in_frames": frames,
                    "transition": "opacity_crossfade",
                    "reason": (
                        "story_beat_change"
                        if str(outgoing.get("renderer") or "") == "video_clip"
                        else "legacy_image_exit"
                    ),
                }
            )
        else:
            outgoing_beat = _package_scene_index(outgoing)
            incoming_beat = _package_scene_index(incoming)
            hard_cuts.append(
                {
                    **boundary,
                    "reason": (
                        "continuity_cut_same_story_beat"
                        if outgoing_beat is not None and outgoing_beat == incoming_beat
                        else "transition_disabled_or_unannotated"
                    ),
                }
            )

    evidence = {
        "policy": "remotion_editorial_story_beat_transitions_v57",
        "requested_transition_frames": requested_frames,
        "boundary_count": max(0, len(shots) - 1),
        "realized_transition_count": len(transitions),
        "hard_cut_count": len(hard_cuts),
        "transitions": transitions,
        "hard_cuts": hard_cuts,
    }
    if evidence["boundary_count"] >= _MINIMUM_REALIZED_TRANSITIONS:
        if len(transitions) < _MINIMUM_REALIZED_TRANSITIONS:
            raise ValueError(
                f"Remotion realized only {len(transitions)} editorial transitions; "
                f"required at least {_MINIMUM_REALIZED_TRANSITIONS}"
            )
    if len(transitions) + len(hard_cuts) != evidence["boundary_count"]:
        raise ValueError("Remotion transition evidence does not cover every shot boundary")
    return evidence


def install_production_remotion_transition_policy_v57() -> None:
    """Replace image-only transition evidence with ViMax story-beat transition evidence."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import production_transition_evidence_v48 as evidence_v48

    evidence_v48.build_remotion_transition_evidence_v48 = (
        build_remotion_editorial_transition_evidence_v57
    )
    _INSTALLED = True
