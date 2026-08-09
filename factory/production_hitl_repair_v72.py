from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .image_generator import KeyframeAsset


class HitlRepairSeedError(RuntimeError):
    """Fail closed when a cross-checkpoint repair request is not exact."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HitlRepairSeedError(f"invalid repair JSON {Path(path).name}: {exc}") from exc
    if not isinstance(value, dict):
        raise HitlRepairSeedError(f"repair JSON must be an object: {Path(path).name}")
    return value


def _parse_rejected_shots(raw: str) -> tuple[int, ...]:
    try:
        values = sorted({int(item.strip()) for item in str(raw).split(",") if item.strip()})
    except ValueError as exc:
        raise HitlRepairSeedError("HITL_REPAIR_REJECTED_SHOTS must contain comma-separated integers") from exc
    if not values or values[0] < 0:
        raise HitlRepairSeedError("repair requires at least one non-negative rejected shot ID")
    return tuple(values)


def _verify_rejected_decision(
    source_dir: Path,
    *,
    approval_subject_sha256: str,
    source_code_sha: str,
    rejected_shots: tuple[int, ...],
) -> dict[str, Any]:
    from .production_hitl_checkpoint_v71 import (
        _DECISION_SCHEMA,
        _canonical_digest,
        verify_hitl_checkpoint_v71,
    )

    checkpoint = verify_hitl_checkpoint_v71(
        source_dir,
        approved_keyframe_sha256=approval_subject_sha256,
        approved_code_sha=source_code_sha,
    )
    decision = _read_json(source_dir / "hitl-decision.json")
    if decision.get("schema_version") != _DECISION_SCHEMA:
        raise HitlRepairSeedError("repair source has no supported HITL decision")
    stored = str(decision.get("decision_sha256") or "").strip().lower()
    unsigned = dict(decision)
    unsigned.pop("decision_sha256", None)
    if not stored or stored != _canonical_digest(unsigned):
        raise HitlRepairSeedError("repair source HITL decision digest is invalid")
    if str(decision.get("canary_id") or "") != source_dir.name:
        raise HitlRepairSeedError("repair decision targets a different canary")
    if str(decision.get("code_sha") or "") != str(checkpoint.get("code_sha") or ""):
        raise HitlRepairSeedError("repair decision targets a different source-code revision")
    if str(decision.get("approval_subject_sha256") or "") != str(checkpoint.get("approval_subject_sha256") or ""):
        raise HitlRepairSeedError("repair decision targets a different sealed checkpoint")
    if str(decision.get("verdict") or "").strip().lower() != "reject":
        raise HitlRepairSeedError("cross-checkpoint repair requires an explicit rejected HITL decision")
    if str(decision.get("reviewer_kind") or "") not in {"human", "human_simulation"}:
        raise HitlRepairSeedError("repair decision reviewer kind is invalid")
    expected = list(range(int(checkpoint.get("expected_shots") or 0)))
    try:
        reviewed = sorted({int(value) for value in decision.get("reviewed_shot_ids") or []})
    except (TypeError, ValueError) as exc:
        raise HitlRepairSeedError("repair decision contains invalid reviewed shot IDs") from exc
    if reviewed != expected:
        raise HitlRepairSeedError("repair source was not explicitly reviewed across every sealed keyframe")
    if any(index not in reviewed for index in rejected_shots):
        raise HitlRepairSeedError("repair request names a shot outside the reviewed checkpoint")
    return {"checkpoint": checkpoint, "decision": decision}


def _normalized_direction(value: str) -> str:
    from .production_visual_semantic_review_v28 import _extract_direction, _normalize_visual_intent

    return " ".join(_normalize_visual_intent(_extract_direction(str(value))).casefold().split())


def load_hitl_repair_seed_v72(
    *,
    plan: Any,
    current_scenes: dict[int, Any],
    output_dir: Path,
) -> tuple[dict[int, KeyframeAsset], dict[str, Any]]:
    """Materialize non-rejected, unchanged-direction frames from an exact rejected checkpoint.

    This function does not approve them. The v28 semantic reviewer must review every returned frame
    against the *current* exact claim before it enters the approved cache.
    """
    source_raw = os.getenv("HITL_REPAIR_SOURCE_DIR", "").strip()
    if not source_raw:
        return {}, {"enabled": False}
    source_dir = Path(source_raw)
    if not source_dir.is_dir():
        raise HitlRepairSeedError("HITL repair source directory does not exist")
    approval_subject = os.getenv("HITL_REPAIR_APPROVAL_SUBJECT_SHA256", "").strip()
    source_code_sha = os.getenv("HITL_REPAIR_SOURCE_CODE_SHA", "").strip().lower()
    rejected = _parse_rejected_shots(os.getenv("HITL_REPAIR_REJECTED_SHOTS", ""))
    verified = _verify_rejected_decision(
        source_dir,
        approval_subject_sha256=approval_subject,
        source_code_sha=source_code_sha,
        rejected_shots=rejected,
    )
    checkpoint = verified["checkpoint"]
    if int(checkpoint.get("expected_shots") or 0) != len(plan.scenes):
        # Changed editorial topology means positional frame reuse is unsafe; generate the new set.
        return {}, {
            "enabled": True,
            "source_canary_id": source_dir.name,
            "source_code_sha": source_code_sha,
            "source_approval_subject_sha256": approval_subject,
            "rejected_shots": list(rejected),
            "eligible_shots": [],
            "skipped_reason": "shot_count_changed",
        }

    manifest = _read_json(source_dir / "keyframe-manifest.json")
    raw_assets = manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
    by_index = {
        int(item.get("scene_index")): item
        for item in raw_assets
        if isinstance(item, dict) and str(item.get("scene_index", "")).lstrip("-").isdigit()
    }
    candidates: dict[int, KeyframeAsset] = {}
    eligible: list[int] = []
    direction_changed: list[int] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in sorted(current_scenes):
        if index in rejected:
            continue
        meta = by_index.get(index)
        if meta is None:
            continue
        old_director = str(meta.get("director_prompt") or "")
        if not old_director or _normalized_direction(old_director) != _normalized_direction(current_scenes[index].image_prompt):
            direction_changed.append(index)
            continue
        if int(meta.get("width") or 0) != int(plan.width) or int(meta.get("height") or 0) != int(plan.height):
            direction_changed.append(index)
            continue
        source = source_dir / "visual-keyframes" / f"scene-{index:02d}-keyframe.png"
        if not source.is_file():
            raise HitlRepairSeedError(f"sealed repair keyframe is missing for shot {index}")
        expected_hash = str((checkpoint.get("keyframes") or {}).get(source.name) or "")
        if not expected_hash or str(meta.get("sha256") or "") != expected_hash:
            raise HitlRepairSeedError(f"repair manifest/hash mismatch for shot {index}")
        destination = Path(output_dir) / source.name
        shutil.copy2(source, destination)
        candidate = KeyframeAsset(
            scene_index=index,
            path=destination,
            model=str(meta.get("model") or ""),
            seed=int(meta.get("seed") or 0),
            width=int(meta.get("width") or 0),
            height=int(meta.get("height") or 0),
            sha256=expected_hash,
            entropy=float(meta.get("entropy") or 0.0),
            prompt=str(meta.get("prompt") or ""),
            negative_prompt=str(meta.get("negative_prompt") or ""),
            director_prompt=old_director,
            prompt_word_count=int(meta.get("prompt_word_count") or 0),
            prompt_word_budget=int(meta.get("prompt_word_budget") or 0),
            prompt_compiler_version=str(meta.get("prompt_compiler_version") or ""),
            caption_zone_detail_before=float(meta.get("caption_zone_detail_before") or 0.0),
            caption_zone_detail_after=float(meta.get("caption_zone_detail_after") or 0.0),
            caption_zone_repaired=bool(meta.get("caption_zone_repaired", False)),
        )
        candidates[index] = candidate
        eligible.append(index)
    return candidates, {
        "enabled": True,
        "source_canary_id": source_dir.name,
        "source_code_sha": source_code_sha,
        "source_approval_subject_sha256": approval_subject,
        "source_decision_sha256": str(verified["decision"].get("decision_sha256") or ""),
        "rejected_shots": list(rejected),
        "eligible_shots": eligible,
        "direction_changed_shots": direction_changed,
    }
