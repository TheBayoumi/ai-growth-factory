from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence

from .production_voice_capacity_v29 import split_narration_for_voice_v29


_INSTALLED = False
_CONNECTORS = {"and", "but", "because", "while", "which", "that", "so", "as", "by", "before", "after"}
_FAILURE_DIAGNOSTICS = (
    "voice-calibration-failure.json",
    "voice-micro-clause-failure.json",
)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _normalized(text: str) -> str:
    return " ".join(text.split()).strip()


def _balanced_exact_split_v50(
    text: str,
    *,
    minimum_words: int,
    maximum_words: int,
) -> tuple[str, str] | None:
    words = _normalized(text).split()
    total = len(words)
    low = max(minimum_words, total - maximum_words)
    high = min(maximum_words, total - minimum_words)
    if low > high:
        return None

    midpoint = total / 2.0
    candidates: list[tuple[int, float, int]] = []
    for index in range(low, high + 1):
        previous = words[index - 1]
        current = words[index].strip(".,;:!?()[]{}\"'").casefold()
        punctuation_priority = 0 if previous.endswith((".", "!", "?", ",", ";", ":")) else 1
        connector_priority = 0 if current in _CONNECTORS else 1
        candidates.append((punctuation_priority + connector_priority, abs(index - midpoint), index))
    split_at = min(candidates)[2]
    left = " ".join(words[:split_at])
    right = " ".join(words[split_at:])
    if _normalized(f"{left} {right}") != _normalized(text):
        raise ValueError("Balanced voice segmentation changed the supplied transcript")
    return left, right


def repair_short_voice_orphans_v50(
    segments: Sequence[str],
    *,
    minimum_words: int = 12,
    maximum_words: int = 24,
    merge_overflow_words: int = 2,
) -> list[str]:
    """Eliminate subminimum Qwen segments without weakening the publication pace gate.

    A short standalone segment pays Qwen's fixed onset/ending cadence and can be structurally
    unreachable even when its delivery is brisk. Preserve sentence boundaries first by merging
    the orphan with the smaller adjacent segment, allowing only a tiny generator-only overflow.
    If neither neighbor fits that bound, repartition one adjacent pair transcript-exactly into two
    balanced pieces that both satisfy the original minimum/maximum word contract.
    """
    if minimum_words < 1 or maximum_words < minimum_words:
        raise ValueError("Voice orphan-repair word limits are inconsistent")
    if not 0 <= merge_overflow_words <= 4:
        raise ValueError("merge_overflow_words must be between zero and four")

    repaired = [_normalized(item) for item in segments if _normalized(item)]
    if not repaired:
        raise ValueError("Voice orphan repair received no segments")
    if len(repaired) == 1:
        return repaired

    original = _normalized(" ".join(repaired))
    index = 0
    while index < len(repaired):
        if len(repaired[index].split()) >= minimum_words:
            index += 1
            continue

        merge_limit = maximum_words + merge_overflow_words
        merge_options: list[tuple[int, int, str]] = []
        if index > 0:
            combined = _normalized(f"{repaired[index - 1]} {repaired[index]}")
            count = len(combined.split())
            if count <= merge_limit:
                merge_options.append((count, index - 1, combined))
        if index + 1 < len(repaired):
            combined = _normalized(f"{repaired[index]} {repaired[index + 1]}")
            count = len(combined.split())
            if count <= merge_limit:
                merge_options.append((count, index, combined))

        if merge_options:
            _count, start, combined = min(merge_options, key=lambda item: (item[0], item[1]))
            repaired[start : start + 2] = [combined]
            index = max(0, start - 1)
            continue

        rebalance_options: list[tuple[int, int, tuple[str, str]]] = []
        for start in (index - 1, index):
            if start < 0 or start + 1 >= len(repaired):
                continue
            combined = _normalized(f"{repaired[start]} {repaired[start + 1]}")
            split = _balanced_exact_split_v50(
                combined,
                minimum_words=minimum_words,
                maximum_words=maximum_words,
            )
            if split is None:
                continue
            imbalance = abs(len(split[0].split()) - len(split[1].split()))
            rebalance_options.append((imbalance, start, split))
        if not rebalance_options:
            raise ValueError(
                f"Voice segmentation left an unrecoverable {len(repaired[index].split())}-word "
                f"orphan at index {index}"
            )
        _imbalance, start, split = min(rebalance_options, key=lambda item: (item[0], item[1]))
        repaired[start : start + 2] = list(split)
        index = max(0, start - 1)

    if any(len(item.split()) < minimum_words for item in repaired):
        raise ValueError("Voice orphan repair left a subminimum segment")
    if any(len(item.split()) > maximum_words + merge_overflow_words for item in repaired):
        raise ValueError("Voice orphan repair exceeded the bounded generator overflow")
    if _normalized(" ".join(repaired)) != original:
        raise ValueError("Voice orphan repair changed the supplied transcript")
    return repaired


def split_narration_for_voice_v50(text: str, target_segments: int) -> list[str]:
    base = split_narration_for_voice_v29(text, target_segments)
    minimum_words = _env_int("V28_MIN_VOICE_SEGMENT_WORDS", 12, 3, 20)
    maximum_words = _env_int("V28_MAX_VOICE_SEGMENT_WORDS", 24, minimum_words, 40)
    overflow = _env_int("V50_ORPHAN_MERGE_OVERFLOW_WORDS", 2, 0, 4)
    return repair_short_voice_orphans_v50(
        base,
        minimum_words=minimum_words,
        maximum_words=maximum_words,
        merge_overflow_words=overflow,
    )


def copy_voice_failure_diagnostics_v50(workdir: Path, destination: Path) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in _FAILURE_DIAGNOSTICS:
        source = workdir / name
        if not source.is_file():
            continue
        target = destination / name
        shutil.copy2(source, target)
        copied.append(target)
    return tuple(copied)


def install_production_voice_orphan_recovery_v50() -> None:
    """Install transcript-exact orphan repair and persist fail-closed voice diagnostics."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import canary, voice_pipeline

    voice_pipeline.split_narration = split_narration_for_voice_v50
    current_copy_voice_diagnostics = canary._copy_voice_diagnostics

    def copy_voice_diagnostics_v50(workdir: Path, destination: Path) -> None:
        current_copy_voice_diagnostics(workdir, destination)
        copy_voice_failure_diagnostics_v50(workdir, destination)

    canary._copy_voice_diagnostics = copy_voice_diagnostics_v50
    os.environ.setdefault("V50_ORPHAN_MERGE_OVERFLOW_WORDS", "2")
    _INSTALLED = True
