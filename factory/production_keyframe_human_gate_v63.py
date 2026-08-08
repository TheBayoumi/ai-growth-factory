from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


_INSTALLED = False


class HumanKeyframeReviewRequired(RuntimeError):
    """Intentional release pause after machine-reviewed keyframes and before temporal inference."""


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _preview_only() -> bool:
    return os.getenv("HITL_KEYFRAME_PREVIEW_ONLY", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _keyframe_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("scene-*-keyframe.png"))


def _contact_sheet(paths: Sequence[Path], output: Path, *, columns: int = 4) -> Path | None:
    paths = [Path(path) for path in paths if Path(path).is_file()]
    if not paths:
        return None
    thumb_w, thumb_h = 256, 456
    label_h = 30
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            frame = ImageOps.fit(source.convert("RGB"), (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(frame, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(18, 18, 20))
        draw.text((x + 8, y + thumb_h + 8), f"SHOT {index:02d}", fill=(240, 240, 240), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    return output


def build_keyframe_human_review_dossier_v63(
    *,
    plan: Any,
    output_dir: Path,
    assets: Sequence[Any] | None = None,
    machine_error: str = "",
) -> dict[str, Any]:
    """Prepare the exact storyboard/keyframes for a senior-editor decision before Wan spend."""
    output_dir = Path(output_dir)
    paths = _keyframe_paths(output_dir)
    manifest = _read_json(output_dir / "keyframe-manifest.json")
    manifest_assets = manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
    asset_by_index = {
        int(item.get("scene_index")): item
        for item in manifest_assets
        if isinstance(item, dict) and str(item.get("scene_index", "")).lstrip("-").isdigit()
    }
    realized = {int(path.stem.split("-")[1]): path for path in paths}
    if assets:
        for item in assets:
            try:
                realized[int(item.scene_index)] = Path(item.path)
            except (AttributeError, TypeError, ValueError):
                continue

    scene_records: list[dict[str, Any]] = []
    plan_scenes = list(getattr(plan, "scenes", ()) or ())
    for index, scene in enumerate(plan_scenes):
        path = realized.get(index)
        meta = asset_by_index.get(index, {})
        scene_records.append(
            {
                "shot_id": index,
                "keyframe": str(path.name) if path and path.is_file() else None,
                "keyframe_sha256": _sha256(path) if path and path.is_file() else None,
                "machine_model": str(meta.get("model") or ""),
                "compiled_prompt": str(meta.get("prompt") or ""),
                "director_prompt": str(getattr(scene, "image_prompt", "")),
                "motion_prompt": str(getattr(scene, "motion_prompt", "")),
                "duration_seconds": float(getattr(scene, "duration_seconds", 0.0) or 0.0),
                "generation_mode": str(getattr(scene, "generation_mode", "")),
                "review": {
                    "semantic_relevance": "pending_human",
                    "subject_and_action": "pending_human",
                    "visual_quality": "pending_human",
                    "continuity": "pending_human",
                    "accept_for_temporal_generation": None,
                    "notes": [],
                },
            }
        )

    all_present = bool(scene_records) and all(item["keyframe_sha256"] for item in scene_records)
    machine_passed = not machine_error and all_present
    status = "awaiting_human_keyframe_review" if machine_passed else "blocked_machine_keyframe_review"
    return {
        "schema_version": "human-keyframe-review-v63",
        "status": status,
        "release_decision": "blocked_pending_human_keyframe_review",
        "machine_keyframe_review_passed": machine_passed,
        "machine_error": machine_error or None,
        "expected_shots": len(plan_scenes),
        "realized_keyframes": sum(item["keyframe_sha256"] is not None for item in scene_records),
        "contact_sheet": "keyframe-contact-sheet.jpg" if paths else None,
        "keyframe_manifest": "keyframe-manifest.json" if manifest else None,
        "human_review_required": True,
        "human_checklist": [
            "Reject generic filler even if it is technically related to AI or servers.",
            "Reject literalization of ambiguous company/product names into fantasy, logos, text, or characters.",
            "Every shot must visibly represent the planned subject, environment, and physical action.",
            "The 20-shot set must have meaningful composition/action diversity, not repeated people-at-monitors imagery.",
            "Opening shots must establish the story immediately and look strong enough to animate into the first ten seconds.",
            "Continuity should feel intentional across adjacent shots without making every frame visually identical.",
            "Reject malformed anatomy, pseudo-text, collage layouts, implausible equipment, or low-quality generations.",
        ],
        "shots": scene_records,
        "human_verdict": None,
        "human_notes": [],
    }


def _write_dossier(plan: Any, output_dir: Path, *, assets: Sequence[Any] | None = None, machine_error: str = "") -> Path:
    root = Path(output_dir).parent
    paths = _keyframe_paths(Path(output_dir))
    _contact_sheet(paths, root / "keyframe-contact-sheet.jpg")
    payload = build_keyframe_human_review_dossier_v63(
        plan=plan,
        output_dir=Path(output_dir),
        assets=assets,
        machine_error=machine_error,
    )
    path = root / "keyframe-human-review-dossier.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def install_production_keyframe_human_gate_v63() -> None:
    """Pause the ViMax canary after approved keyframes, before any Wan clip generation."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import canary, visual_pipeline

    current_generate = visual_pipeline.generate_keyframes
    if not getattr(current_generate, "_agf_v63", False):
        def generate_keyframes_v63(plan: Any, output_dir: Path) -> Any:
            if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
                return current_generate(plan, output_dir)
            try:
                assets = current_generate(plan, output_dir)
            except Exception as exc:
                _write_dossier(plan, output_dir, machine_error=str(exc))
                raise
            dossier = _write_dossier(plan, output_dir, assets=assets)
            if _preview_only():
                raise HumanKeyframeReviewRequired(
                    "Machine-reviewed ViMax keyframes are ready for mandatory human review before temporal generation; "
                    f"dossier={dossier.name}"
                )
            return assets

        generate_keyframes_v63._agf_v63 = True  # type: ignore[attr-defined]
        visual_pipeline.generate_keyframes = generate_keyframes_v63

    current_copy = canary._copy_visual_audit
    if not getattr(current_copy, "_agf_v63", False):
        def copy_visual_audit_v63(workdir: Path, destination: Path) -> None:
            current_copy(workdir, destination)
            visual_root = Path(workdir) / "visual-assets"
            for name in ("keyframe-human-review-dossier.json", "keyframe-contact-sheet.jpg"):
                source = visual_root / name
                if source.is_file():
                    shutil.copy2(source, Path(destination) / name)

        copy_visual_audit_v63._agf_v63 = True  # type: ignore[attr-defined]
        canary._copy_visual_audit = copy_visual_audit_v63

    _INSTALLED = True
