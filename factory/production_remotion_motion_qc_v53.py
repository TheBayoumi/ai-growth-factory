from __future__ import annotations

from typing import Any, Sequence


_INSTALLED = False


def remotion_motion_failures_v53(
    media_types: Sequence[str],
    report: Any,
) -> tuple[str, ...]:
    """Detect static frames and real discontinuities without treating smooth zoom as a jump.

    The legacy ``jump_ratio`` is the fraction of frame deltas above 0.90. For a smooth
    high-detail Remotion transform that ratio is expected to approach 1.0, so it is motion
    density, not evidence of a discontinuity. A true pixel jump instead produces an isolated
    maximum far above the window mean.
    """
    failures: list[str] = []
    for index, (media_type, window_mean, near_static, jump_ratio, maximum) in enumerate(
        zip(
            media_types,
            report.temporal_window_mean_differences,
            report.temporal_window_near_static_ratios,
            report.temporal_window_jump_ratios,
            report.temporal_window_max_differences,
            strict=False,
        )
    ):
        if media_type == "image":
            if window_mean < 0.08 or near_static > 0.45:
                failures.append(f"image shot {index} is effectively static after composition")
                continue

            # Continuous Remotion moves in the reviewed real canary reached mean~3.62 and
            # max~6.11 while remaining visually smooth. Keep a hard continuous-motion ceiling
            # above that measured envelope, and separately reject isolated discontinuities.
            discontinuity_ratio = maximum / max(float(window_mean), 0.01)
            if window_mean > 4.50 or (
                maximum > 2.80
                and discontinuity_ratio > 3.20
                and jump_ratio < 0.70
            ):
                failures.append(f"image shot {index} has excessive camera motion")
        elif media_type == "video":
            if window_mean < 0.12 or near_static > 0.75:
                failures.append(f"Wan shot {index} has no meaningful visible motion")
            if maximum > 4.50:
                failures.append(f"Wan shot {index} has unstable frame-to-frame motion")
    return tuple(failures)


def install_production_remotion_motion_qc_v53() -> None:
    """Use the reviewed Remotion-aware motion gate without weakening freeze/Wan checks."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_video_qc

    production_video_qc.production_motion_failures = remotion_motion_failures_v53
    _INSTALLED = True
