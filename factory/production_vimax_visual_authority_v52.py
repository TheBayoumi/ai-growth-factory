from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence


_INSTALLED = False
_MAX_WORDS = 58
_MAX_NEGATIVE_WORDS = 36
_MAX_PLAN_ATTEMPTS = 2

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
_SHOT_INDEX_RE = re.compile(r"\[VIMAX_SHOT_INDEX=(\d+)\]", re.IGNORECASE)
_REPAIR_RE = re.compile(r"(?:V28 REPAIR|REQUIRED CORRECTION):\s*(.+)$", re.IGNORECASE)
_TAIL_RE = re.compile(
    r"\s*(?:\.\s*)?(?:V30 STORYBOARD|V31 REPAIR|V28 REPAIR|REQUIRED CORRECTION):.*$",
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(
    r"Supporting source-grounded visual direction:\s*(.+?)\.\s*Shot treatment:",
    re.IGNORECASE,
)
_TEXT_UI_RE = re.compile(
    r"\b(?:shot of the text|text is displayed|text-based information|digital interface|"
    r"interface displays|interface remains|readable text|headline|web page|model page|"
    r"dashboard|screen content)\b",
    re.IGNORECASE,
)
_STATIC_RE = re.compile(
    r"\b(?:static camera|no significant changes?|no changes?|remains in the same|"
    r"no movement|no camera movement|unchanged composition)\b",
    re.IGNORECASE,
)

_FALLBACK_SETUPS = (
    "a developer workstation beside compact inference compute with abstract unreadable model tiles",
    "a developer integration bench with a secure hardware key, unbranded workstation, and compact compute appliance",
    "a software-infrastructure lab where one developer routes a model request through several unbranded compute nodes",
    "a developer test bench linking an unbranded workstation, application device, and compact inference server",
    "two engineers validating one model-serving workflow beside a short compute rack",
    "an integration workspace with one developer, an unbranded workstation, and abstract unreadable model-selection cards",
)

_NEGATIVE_ITEMS = (
    "readable text lettering numbers",
    "logo watermark trademark",
    "collage split frame",
    "pseudo text gibberish typography",
    "malformed anatomy distorted hands",
    "duplicate people duplicate objects",
    "warped equipment",
    "camera shake flicker",
    "black and white monochrome archival photograph",
    "generic robot unrelated machinery",
)


class ViMaxVisualAuthorityError(ValueError):
    """Raised before media inference when ViMax collapses into text cards or static repeats."""


def _enabled() -> bool:
    return os.getenv("VIMAX_PLANNER_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip(" ,.;:")


def _words(value: str) -> list[str]:
    return _TOKEN_RE.findall(_clean(value))


def _fit(parts: Sequence[str], *, limit: int) -> str:
    result: list[str] = []
    for part in parts:
        for word in _words(part):
            if len(result) >= limit:
                return " ".join(result).strip(" ,.;:") + "."
            result.append(word)
    return " ".join(result).strip(" ,.;:") + "."


def _strip_adapter_tail(value: str) -> str:
    return _clean(_TAIL_RE.sub("", value))


def _scene_index(value: str) -> int:
    match = _SHOT_INDEX_RE.search(value)
    return int(match.group(1)) if match else 0


def _quoted_semantic_core(value: str) -> str:
    quoted = [
        _clean(item)
        for item in re.findall(r"['\"]([^'\"]{12,})['\"]", value)
        if _clean(item)
    ]
    if quoted:
        return max(quoted, key=lambda item: len(_words(item)))
    about = re.search(
        r"\babout\s+(.+?)(?:\.\s+The\s+(?:interface|camera|background)|$)",
        value,
        flags=re.IGNORECASE,
    )
    if about:
        return _clean(about.group(1))
    return ""


def _camera_hint(value: str) -> str:
    lowered = value.casefold()
    size = "medium"
    if "close-up" in lowered or "close up" in lowered:
        size = "close"
    elif "wide" in lowered or "establish" in lowered:
        size = "wide"
    angle = "eye-level"
    if "high angle" in lowered or "high-angle" in lowered:
        angle = "high-angle"
    elif "low angle" in lowered or "low-angle" in lowered:
        angle = "low-angle"
    elif "top-down" in lowered or "top down" in lowered:
        angle = "top-down"
    return f"{size} {angle} documentary framing"


def _raw_vimax_direction(value: str) -> str:
    base = _strip_adapter_tail(value)
    match = _DIRECTION_RE.search(base)
    if match:
        return _clean(match.group(1))
    return _clean(_SHOT_INDEX_RE.sub("", base))


def _semantic_direction(value: str) -> tuple[str, bool]:
    direction = _raw_vimax_direction(value)
    textual = bool(_TEXT_UI_RE.search(direction))
    if textual:
        core = _quoted_semantic_core(direction)
        if not core:
            core = re.sub(
                r"\b(?:digital interface|interface|screen|text|text-based information|"
                r"displayed|displays|background|slight(?:ly)? blurred|soft glow)\b",
                " ",
                direction,
                flags=re.IGNORECASE,
            )
            core = _clean(core)
        return core or "source-grounded AI inference integration", True

    direction = re.sub(
        r"\bVertical 9:16 photorealistic technology documentary frame\b.*$",
        "",
        direction,
        flags=re.IGNORECASE,
    )
    return _clean(direction), False


def _negative_prompt() -> str:
    return _fit(_NEGATIVE_ITEMS, limit=_MAX_NEGATIVE_WORDS).rstrip(".")


def compile_vimax_image_prompt_v52(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = _MAX_WORDS,
) -> Any:
    """Compile the ViMax scene itself; never substitute a legacy static frame bank."""
    del director_negative_prompt
    from .visual_prompt_compiler import CompiledVisualPrompt

    budget = min(_MAX_WORDS, max(44, int(word_budget)))
    direction, textual = _semantic_direction(director_prompt)
    index = _scene_index(director_prompt)
    repair_match = _REPAIR_RE.search(director_prompt)
    repair = _clean(repair_match.group(1)) if repair_match else ""

    if textual:
        positive = _fit(
            (
                "Photorealistic vertical technology documentary still",
                direction,
                _FALLBACK_SETUPS[index % len(_FALLBACK_SETUPS)],
                "show the software claim through a concrete developer workflow with no readable interface content",
                _camera_hint(director_prompt),
                repair,
            ),
            limit=budget,
        )
    else:
        positive = _fit(
            (
                "Photorealistic vertical technology documentary still",
                direction,
                "single coherent full-frame scene with realistic anatomy materials and equipment",
                "natural color documentary lighting with restrained blue and warm amber accents",
                _camera_hint(director_prompt),
                repair,
            ),
            limit=budget,
        )

    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=positive,
        negative_prompt=_negative_prompt(),
        word_count=len(_words(positive)),
        word_budget=budget,
        compiler_version="visual-compiler-v52-vimax-authority",
    )


def scene_for_attempt_v52(
    scene: Any,
    *,
    scene_index: int,
    attempt: int,
    repair: str = "",
) -> Any:
    """Retry the same ViMax scene; reviewer feedback may refine it but not replace its subject."""
    base = _strip_adapter_tail(str(scene.image_prompt))
    suffix = ""
    if repair:
        bounded = " ".join(_words(repair)[:20])
        suffix = f". V28 REPAIR: {bounded}"
    return replace(
        scene,
        image_prompt=base + suffix,
        negative_prompt=_negative_prompt(),
        seed=(int(scene.seed) + 32452843 * max(0, attempt - 1)) & 0x7FFFFFFF,
    )


def _normalized_signature(value: str) -> str:
    value = _strip_adapter_tail(value).casefold()
    value = re.sub(
        r"\b(?:close-up|close up|medium shot|wide shot|high angle|high-angle|"
        r"eye level|eye-level|vertical 9:16|soft glow|slightly blurred)\b",
        " ",
        value,
    )
    return _clean(value)


def validate_vimax_storyboard_v52(plan: Any) -> None:
    """Reject cheap-to-fix planner collapse before image/video inference consumes GPU budget."""
    scenes = list(getattr(plan, "scenes", ()) or ())
    if not scenes:
        raise ViMaxVisualAuthorityError("ViMax returned no scenes")

    directions = [_raw_vimax_direction(str(scene.image_prompt)) for scene in scenes]
    textual = [index for index, value in enumerate(directions) if _TEXT_UI_RE.search(value)]
    allowed_textual = min(2, max(0, len(scenes) // 8))
    if len(textual) > allowed_textual:
        raise ViMaxVisualAuthorityError(
            f"ViMax storyboard uses text/interface cards in {len(textual)}/{len(scenes)} shots; "
            f"allowed at most {allowed_textual}: {textual}"
        )

    signatures = Counter(_normalized_signature(value) for value in directions)
    repeated = {key: count for key, count in signatures.items() if key and count > 2}
    if repeated:
        raise ViMaxVisualAuthorityError(
            "ViMax storyboard repeats the same visual direction more than twice: "
            + str(repeated)
        )

    static_only = [
        index
        for index, scene in enumerate(scenes)
        if _STATIC_RE.search(str(scene.motion_prompt))
    ]
    if len(static_only) > max(2, len(scenes) // 8):
        raise ViMaxVisualAuthorityError(
            f"ViMax storyboard is effectively static in {len(static_only)}/{len(scenes)} shots: "
            f"{static_only}"
        )


def _enrich_from_vimax_artifact(plan: Any, package: Any) -> Any:
    from . import vimax_planner

    if not str(getattr(plan, "prompt_version", "")).startswith("vimax-script2video@"):
        return plan
    artifact = vimax_planner._PLAN_ARTIFACTS.get(plan.director_input_sha256)
    if artifact is None or not Path(artifact).is_file():
        raise ViMaxVisualAuthorityError("ViMax planning artifact is unavailable for semantic enrichment")
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    raw_shots = payload.get("shot_descriptions")
    if not isinstance(raw_shots, list) or len(raw_shots) != len(plan.scenes):
        raise ViMaxVisualAuthorityError("ViMax planning artifact does not match the visual plan")

    package_scenes = list(package.scenes)
    if not package_scenes:
        raise ViMaxVisualAuthorityError("Factory package contains no source-grounded scenes")

    enriched = []
    for index, (scene, raw) in enumerate(zip(plan.scenes, raw_shots, strict=True)):
        if not isinstance(raw, dict):
            raise ViMaxVisualAuthorityError(f"ViMax shot {index} is not an object")
        visual = _clean(raw.get("visual_desc"))
        first = _clean(raw.get("ff_desc"))
        if not visual or not first:
            raise ViMaxVisualAuthorityError(f"ViMax shot {index} lost visual/first-frame semantics")
        package_index = min(len(package_scenes) - 1, index * len(package_scenes) // len(plan.scenes))
        claim = _clean(package_scenes[package_index].body)
        if not claim:
            claim = _clean(package_scenes[package_index].heading)
        prompt = (
            f"[VIMAX_SHOT_INDEX={index}] "
            f"Factual technology documentary shot synchronized to this exact spoken sentence: {claim}. "
            f"Supporting source-grounded visual direction: {visual}. "
            f"Shot treatment: {_camera_hint(first)}. "
            f"ViMax first frame: {first}."
        )
        enriched.append(replace(scene, image_prompt=prompt))
    return replace(plan, scenes=tuple(enriched))


def _discard_plan_artifact(plan: Any) -> None:
    from . import vimax_planner

    path = vimax_planner._PLAN_ARTIFACTS.pop(getattr(plan, "director_input_sha256", ""), None)
    if path is not None:
        shutil.rmtree(Path(path).parent, ignore_errors=True)


def _wrap_request_payload() -> None:
    from . import vimax_planner

    current = vimax_planner._request_payload
    if getattr(current, "_agf_v52", False):
        return

    def request_payload_v52(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = current(*args, **kwargs)
        extra = (
            " STORYBOARD QUALITY OVERRIDE: Never visualize narration as written text, title cards, "
            "web pages, dashboards, or a digital interface as the primary subject. Translate software "
            "claims into concrete developer actions, model-serving workflows, compute interactions, "
            "or other physically filmable consequences. At most two shots may contain a secondary "
            "unreadable screen. No environment/subject/action composition may repeat more than twice. "
            "At most two shots may be static: every other shot needs visible subject motion, environment "
            "motion, or a controlled camera move. Vary shot size and controlled movement across the film "
            "(push, pull, pan, tilt, track, reveal) while preserving continuity and avoiding shake. "
            "Do not repeat narration verbatim in visual_desc, ff_desc, lf_desc, or motion_desc."
        )
        payload["user_requirement"] = _clean(str(payload["user_requirement"]) + extra)
        return payload

    request_payload_v52._agf_v52 = True  # type: ignore[attr-defined]
    vimax_planner._request_payload = request_payload_v52


def _install_construct_wrapper() -> None:
    from . import visual_prompt

    current = visual_prompt.construct_visual_plan
    if getattr(current, "_agf_v52", False):
        return

    def construct_visual_plan_v52(
        settings: Any,
        package: Any,
        sources: list[Any],
        strategy: Any,
        *,
        plan_validator: Callable[[Any], None] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_PLAN_ATTEMPTS + 1):
            plan = current(
                settings,
                package,
                sources,
                strategy,
                plan_validator=None,
            )
            try:
                plan = _enrich_from_vimax_artifact(plan, package)
                validate_vimax_storyboard_v52(plan)
                if plan_validator is not None:
                    plan_validator(plan)
                return plan
            except Exception as exc:
                last_error = exc
                _discard_plan_artifact(plan)
                if attempt >= _MAX_PLAN_ATTEMPTS:
                    break
                print(
                    f"[vimax-v52] planner quality retry {attempt + 1}/{_MAX_PLAN_ATTEMPTS}: {exc}",
                    flush=True,
                )
        raise ViMaxVisualAuthorityError(
            f"ViMax storyboard failed bounded quality convergence: {last_error}"
        ) from last_error

    construct_visual_plan_v52._agf_v52 = True  # type: ignore[attr-defined]
    visual_prompt.construct_visual_plan = construct_visual_plan_v52
    for module_name in ("factory.pipeline", "factory.canary"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "construct_visual_plan", construct_visual_plan_v52)


def _install_motion_propagation() -> None:
    from . import production_editorial_v28
    from . import production_remotion_renderer_v45 as remotion_v45

    current_editorial = production_editorial_v28.build_editorial_plan
    if not getattr(current_editorial, "_agf_v52", False):
        def build_editorial_plan_v52(*args: Any, **kwargs: Any) -> Any:
            updated_plan, shots = current_editorial(*args, **kwargs)
            scene_by_index = {
                int(scene.scene_index): scene
                for scene in updated_plan.scenes
            }
            updated_shots = []
            for shot in shots:
                scene = scene_by_index[int(shot.shot_id)]
                motion = _clean(scene.motion_prompt)
                updated_shots.append(
                    replace(
                        shot,
                        treatment=_clean(
                            f"{shot.treatment}; VIMAX_MOTION: {motion}"
                        ),
                    )
                )
            return updated_plan, tuple(updated_shots)

        build_editorial_plan_v52._agf_v52 = True  # type: ignore[attr-defined]
        production_editorial_v28.build_editorial_plan = build_editorial_plan_v52

    current_render_spec = remotion_v45.build_remotion_render_spec
    if getattr(current_render_spec, "_agf_v52", False):
        return

    def movement_from_motion(value: str) -> str:
        text = value.casefold()
        for needles, movement in (
            (("dolly out", "pull out", "zoom out", "pulls back"), "dolly_out"),
            (("dolly in", "push in", "pushes in", "zoom in", "moves closer"), "dolly_in"),
            (("pan left", "pans left", "track left", "tracks left"), "pan_left"),
            (("pan right", "pans right", "track right", "tracks right"), "pan_right"),
            (("tilt up", "tilts up"), "tilt_up"),
            (("tilt down", "tilts down"), "tilt_down"),
            (("static camera", "locked camera", "camera remains fixed"), "static"),
        ):
            if any(needle in text for needle in needles):
                return movement
        return "subtle_push_in"

    def render_spec_v52(*args: Any, **kwargs: Any) -> Any:
        source_shots = kwargs.get("shots")
        if source_shots is None and args:
            raise ViMaxVisualAuthorityError("v52 Remotion bridge requires keyword shot arguments")
        source_shots = list(source_shots or ())
        spec = current_render_spec(*args, **kwargs)
        if len(source_shots) != len(spec.shots):
            raise ViMaxVisualAuthorityError("v52 Remotion shot count changed during translation")
        translated = []
        for render_shot, source_shot in zip(spec.shots, source_shots, strict=True):
            treatment = _clean(getattr(source_shot, "treatment", ""))
            match = re.search(r"VIMAX_MOTION:\s*(.+)$", treatment, flags=re.IGNORECASE)
            if not match:
                raise ViMaxVisualAuthorityError(
                    f"shot {render_shot.shot_id} lost ViMax motion before Remotion"
                )
            motion = _clean(match.group(1))
            camera = replace(
                render_shot.camera,
                movement=movement_from_motion(motion),
            )
            translated.append(
                replace(
                    render_shot,
                    motion_prompt=motion,
                    camera=camera,
                )
            )
        updated = replace(spec, shots=tuple(translated))
        updated.validate(require_files=True)
        return updated

    render_spec_v52._agf_v52 = True  # type: ignore[attr-defined]
    remotion_v45.build_remotion_render_spec = render_spec_v52


def install_production_vimax_visual_authority_v52() -> None:
    """Make the live canary use ViMax semantics and motion all the way through Remotion."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    from . import image_generator, visual_prompt_compiler
    from . import production_visual_semantic_review_v28 as semantic_v28

    _wrap_request_payload()
    _install_construct_wrapper()

    visual_prompt_compiler.compile_image_prompt = compile_vimax_image_prompt_v52
    image_generator.compile_image_prompt = compile_vimax_image_prompt_v52
    semantic_v28._scene_for_attempt = scene_for_attempt_v52

    _install_motion_propagation()
    _INSTALLED = True
