from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PINNED_VIMAX_COMMIT = "05a48943878312d88fe5a016c12a9654940ecc43"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViMax Script2Video text planning only")
    parser.add_argument("--vimax-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _item_payload(item: Any) -> dict[str, Any]:
    if item is None:
        return {
            "parent_cam_idx": None,
            "parent_shot_idx": None,
            "reason": "",
            "is_parent_fully_covers_child": None,
            "missing_info": None,
        }
    if hasattr(item, "model_dump"):
        raw = item.model_dump()
    elif isinstance(item, dict):
        raw = dict(item)
    else:
        raw = {
            "parent_cam_idx": getattr(item, "parent_cam_idx", None),
            "parent_shot_idx": getattr(item, "parent_shot_idx", None),
            "reason": getattr(item, "reason", ""),
            "is_parent_fully_covers_child": getattr(
                item, "is_parent_fully_covers_child", None
            ),
            "missing_info": getattr(item, "missing_info", None),
        }
    return {
        "parent_cam_idx": raw.get("parent_cam_idx"),
        "parent_shot_idx": raw.get("parent_shot_idx"),
        "reason": str(raw.get("reason") or ""),
        "is_parent_fully_covers_child": raw.get("is_parent_fully_covers_child"),
        "missing_info": raw.get("missing_info"),
    }


def _nearest_parent_shot(parent_camera: Any, child_camera: Any) -> int:
    parent_shots = [int(value) for value in parent_camera.active_shot_idxs]
    if not parent_shots:
        raise ValueError(f"Camera {parent_camera.idx} has no active shots")
    child_shots = [int(value) for value in child_camera.active_shot_idxs]
    child_anchor = min(child_shots) if child_shots else parent_shots[0]
    return min(parent_shots, key=lambda value: (abs(value - child_anchor), value))


def _find_cycle(parent_by_camera: dict[int, int | None]) -> list[int] | None:
    completed: set[int] = set()
    for start in parent_by_camera:
        if start in completed:
            continue
        order: list[int] = []
        position: dict[int, int] = {}
        current: int | None = start
        while current is not None and current not in completed:
            if current in position:
                return order[position[current] :]
            position[current] = len(order)
            order.append(current)
            current = parent_by_camera.get(current)
        completed.update(order)
    return None


def sanitize_camera_parent_items(
    cameras: Sequence[Any],
    parent_items: Sequence[Any],
) -> tuple[list[dict[str, Any] | None], list[dict[str, Any]]]:
    """Repair only invalid ViMax camera-tree edges and preserve every valid edge."""
    if len(cameras) != len(parent_items):
        raise ValueError(
            f"Camera tree response length mismatch: expected {len(cameras)}, "
            f"got {len(parent_items)}"
        )
    if not cameras:
        raise ValueError("Camera tree requires at least one camera")

    camera_by_idx = {int(camera.idx): camera for camera in cameras}
    camera_order = [int(camera.idx) for camera in cameras]
    if len(camera_by_idx) != len(cameras):
        raise ValueError("Camera indices must be unique")
    if camera_order[0] != 0:
        raise ValueError(f"The first ViMax camera must be camera 0; got {camera_order[0]}")
    order_position = {camera_idx: index for index, camera_idx in enumerate(camera_order)}
    repairs: list[dict[str, Any]] = []
    normalized: dict[int, dict[str, Any] | None] = {}

    def earlier_parent(camera_idx: int, *, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        candidates = [
            candidate
            for candidate in camera_order
            if order_position[candidate] < order_position[camera_idx]
            and candidate not in excluded
        ]
        return candidates[-1] if candidates else 0

    def record(camera_idx: int, issue: str, old_parent: Any, new_parent: Any) -> None:
        repairs.append(
            {
                "camera_idx": camera_idx,
                "issue": issue,
                "old_parent_cam_idx": old_parent,
                "new_parent_cam_idx": new_parent,
            }
        )

    for camera, raw_item in zip(cameras, parent_items, strict=True):
        camera_idx = int(camera.idx)
        payload = _item_payload(raw_item)
        proposed_parent = payload["parent_cam_idx"]
        try:
            proposed_parent = (
                int(proposed_parent) if proposed_parent is not None else None
            )
        except (TypeError, ValueError):
            proposed_parent = None

        if camera_idx == 0:
            if proposed_parent is not None:
                record(camera_idx, "root_camera_had_parent", proposed_parent, None)
            normalized[camera_idx] = None
            continue

        parent_idx = proposed_parent
        repaired_for_camera = False
        if parent_idx is None:
            parent_idx = earlier_parent(camera_idx)
            record(camera_idx, "extra_root", None, parent_idx)
            repaired_for_camera = True
        elif parent_idx == camera_idx:
            parent_idx = earlier_parent(camera_idx)
            record(camera_idx, "self_parent", proposed_parent, parent_idx)
            repaired_for_camera = True
        elif parent_idx not in camera_by_idx:
            parent_idx = earlier_parent(camera_idx)
            record(camera_idx, "missing_parent_camera", proposed_parent, parent_idx)
            repaired_for_camera = True

        parent_camera = camera_by_idx[parent_idx]
        valid_parent_shots = {int(value) for value in parent_camera.active_shot_idxs}
        proposed_parent_shot = payload["parent_shot_idx"]
        try:
            proposed_parent_shot = (
                int(proposed_parent_shot)
                if proposed_parent_shot is not None
                else None
            )
        except (TypeError, ValueError):
            proposed_parent_shot = None
        if proposed_parent_shot not in valid_parent_shots:
            repaired_parent_shot = _nearest_parent_shot(parent_camera, camera)
            repairs.append(
                {
                    "camera_idx": camera_idx,
                    "issue": "invalid_parent_shot",
                    "parent_cam_idx": parent_idx,
                    "old_parent_shot_idx": proposed_parent_shot,
                    "new_parent_shot_idx": repaired_parent_shot,
                }
            )
            proposed_parent_shot = repaired_parent_shot
            repaired_for_camera = True

        payload["parent_cam_idx"] = parent_idx
        payload["parent_shot_idx"] = proposed_parent_shot
        if repaired_for_camera:
            payload["reason"] = (
                f"{payload['reason']} Adapter repaired invalid camera-tree metadata."
            ).strip()
        normalized[camera_idx] = payload

    parent_by_camera = {
        camera_idx: None if payload is None else int(payload["parent_cam_idx"])
        for camera_idx, payload in normalized.items()
    }
    while True:
        cycle = _find_cycle(parent_by_camera)
        if not cycle:
            break
        target = max(cycle, key=lambda value: order_position[value])
        replacement = earlier_parent(target, excluded=set(cycle))
        payload = normalized[target]
        if payload is None:
            raise ValueError("Camera 0 unexpectedly participated in a cycle")
        old_parent = payload["parent_cam_idx"]
        payload["parent_cam_idx"] = replacement
        payload["parent_shot_idx"] = _nearest_parent_shot(
            camera_by_idx[replacement], camera_by_idx[target]
        )
        payload["reason"] = (
            f"{payload['reason']} Adapter broke an invalid camera-tree cycle."
        ).strip()
        parent_by_camera[target] = replacement
        record(target, "cycle", old_parent, replacement)

    roots = [camera_idx for camera_idx, parent in parent_by_camera.items() if parent is None]
    if roots != [0]:
        raise ValueError(f"Sanitized camera tree must have only camera 0 as root; got {roots}")
    for camera_idx in camera_order:
        current = camera_idx
        visited: set[int] = set()
        while current != 0:
            if current in visited:
                raise ValueError(f"Sanitized camera tree still cycles at camera {camera_idx}")
            visited.add(current)
            parent = parent_by_camera.get(current)
            if parent is None:
                raise ValueError(
                    f"Sanitized camera {camera_idx} does not reach root camera 0"
                )
            current = parent

    return [normalized[camera_idx] for camera_idx in camera_order], repairs


def _install_camera_tree_repair() -> None:
    import agents.camera_image_generator as camera_module

    async def construct_camera_tree_repaired(
        self: Any,
        cameras: list[Any],
        shot_descs: list[Any],
    ) -> list[Any]:
        parser = camera_module.PydanticOutputParser(
            pydantic_object=camera_module.CameraTreeResponse
        )
        shot_desc_by_idx = {int(shot.idx): shot for shot in shot_descs}
        camera_seq_str = "<CAMERA_SEQ>\n"
        for camera in cameras:
            camera_seq_str += f"<CAMERA_{camera.idx}>\n"
            for shot_idx in camera.active_shot_idxs:
                shot_desc = shot_desc_by_idx.get(int(shot_idx))
                if shot_desc is None:
                    raise ValueError(
                        f"Camera {camera.idx} references missing shot {shot_idx}"
                    )
                camera_seq_str += f"Shot {shot_idx}: {shot_desc.visual_desc}\n"
            camera_seq_str += f"</CAMERA_{camera.idx}>\n"
        camera_seq_str += "</CAMERA_SEQ>"

        messages = [
            camera_module.SystemMessage(
                content=camera_module.system_prompt_template_select_reference_camera.format(
                    format_instructions=parser.get_format_instructions()
                )
            ),
            camera_module.HumanMessage(
                content=camera_module.human_prompt_template_select_reference_camera.format(
                    camera_seq_str=camera_seq_str
                )
            ),
        ]
        response = await (self.chat_model | parser).ainvoke(messages)
        normalized, repairs = sanitize_camera_parent_items(
            cameras, response.camera_parent_items
        )
        self._agf_camera_tree_repairs = repairs
        for camera, payload in zip(cameras, normalized, strict=True):
            if payload is None:
                camera.parent_cam_idx = None
                camera.parent_shot_idx = None
                camera.reason = None
                camera.is_parent_fully_covers_child = None
                camera.missing_info = None
                continue
            item = camera_module.CameraParentItem.model_validate(payload)
            camera.parent_cam_idx = item.parent_cam_idx
            camera.parent_shot_idx = item.parent_shot_idx
            camera.reason = item.reason
            camera.is_parent_fully_covers_child = item.is_parent_fully_covers_child
            camera.missing_info = item.missing_info
        return cameras

    camera_module.CameraImageGenerator.construct_camera_tree = construct_camera_tree_repaired


async def _run(request: dict[str, Any], *, vimax_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(vimax_root))
    try:
        from langchain.chat_models import init_chat_model
        from pipelines.script2video_pipeline import Script2VideoPipeline
        from utils.provider_presets import resolve_chat_model_config
    except Exception as exc:
        raise RuntimeError(f"could not import pinned ViMax planner: {exc}") from exc

    _install_camera_tree_repair()
    chat_config = request.get("chat_model")
    if not isinstance(chat_config, dict):
        raise ValueError("request.chat_model must be an object")
    chat_model = init_chat_model(**resolve_chat_model_config(dict(chat_config)))
    working_dir = Path(str(request["working_dir"])).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Script2VideoPipeline(
        chat_model=chat_model,
        image_generator=None,
        video_generator=None,
        working_dir=str(working_dir),
    )
    planned = await pipeline.plan_text_artifacts(
        script=str(request["script"]),
        user_requirement=str(request["user_requirement"]),
        style=str(request["style"]),
        progress=None,
        quiet=True,
    )
    repairs = getattr(pipeline.camera_image_generator, "_agf_camera_tree_repairs", [])
    return {
        "status": "planned",
        "schema_version": "agf-vimax-plan-v1",
        "vimax_commit": PINNED_VIMAX_COMMIT,
        "characters": _dump(planned["characters"]),
        "storyboard": _dump(planned["storyboard"]),
        "shot_descriptions": _dump(planned["shot_descriptions"]),
        "camera_tree": _dump(planned["camera_tree"]),
        "camera_tree_repairs": _dump(repairs),
        "working_dir": str(working_dir),
    }


def main() -> None:
    args = _args()
    vimax_root = Path(args.vimax_root).resolve()
    request_path = Path(args.request).resolve()
    output_path = Path(args.output).resolve()
    if not (vimax_root / "pipelines" / "script2video_pipeline.py").is_file():
        raise SystemExit(f"invalid ViMax root: {vimax_root}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("vimax_commit") != PINNED_VIMAX_COMMIT:
        raise SystemExit("request ViMax commit does not match pinned adapter")
    payload = asyncio.run(_run(request, vimax_root=vimax_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
