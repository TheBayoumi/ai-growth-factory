from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import requests

from .config import Settings
from .feeds import SourceItem
from .models import VideoPackage
from .policy import Strategy


PROMPT_VERSION = "visual-director-v1"
_IMAGE_SUFFIX = (
    " Vertical 9:16 composition at 704 by 1280. Keep the lower 32 percent visually "
    "quiet for a separate animated caption layer. No text, letters, numbers, captions, "
    "subtitles, logos, trademarks, watermarks, interface labels, signatures, or borders."
)
_MOTION_SUFFIX = (
    " Preserve the keyframe composition and subject identity. Controlled motion only. "
    "No camera shake, no morphing, no new objects, no disappearing objects, no text, "
    "no logos, and no scene cut."
)
_DEFAULT_NEGATIVE = (
    "text, typography, letters, numbers, captions, subtitles, logo, watermark, signature, "
    "brand mark, UI label, blurry, low resolution, compression artifacts, oversharpened, "
    "deformed objects, duplicated objects, warped geometry, flicker, camera shake, morphing, "
    "new objects, disappearing objects, abrupt cut, cluttered lower third"
)
_FORBIDDEN_IMAGE_DIRECTIVES = (
    "write the words",
    "display the text",
    "caption reads",
    "headline reads",
    "show the logo",
    "include a watermark",
    "add subtitles",
)


class VisualPromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneVisualPrompt:
    scene_index: int
    source_index: int
    role: str
    generation_mode: str
    image_prompt: str
    motion_prompt: str
    negative_prompt: str
    continuity_anchor: str
    caption_safe_zone: str
    seed: int
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualPlan:
    prompt_version: str
    global_style: str
    palette: str
    lighting: str
    continuity_bible: str
    image_model: str
    video_model: str
    width: int
    height: int
    fps: int
    director_input_sha256: str
    scenes: tuple[SceneVisualPrompt, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenes"] = [scene.as_dict() for scene in self.scenes]
        return payload


_VISUAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "global_style": {"type": "string"},
        "palette": {"type": "string"},
        "lighting": {"type": "string"},
        "continuity_bible": {"type": "string"},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_index": {"type": "integer"},
                    "source_index": {"type": "integer"},
                    "role": {"type": "string"},
                    "generation_mode": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "motion_prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "continuity_anchor": {"type": "string"},
                },
                "required": [
                    "scene_index",
                    "source_index",
                    "role",
                    "generation_mode",
                    "image_prompt",
                    "motion_prompt",
                    "negative_prompt",
                    "continuity_anchor",
                ],
            },
        },
    },
    "required": [
        "global_style",
        "palette",
        "lighting",
        "continuity_bible",
        "scenes",
    ],
}


def _extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        raise VisualPromptError("Visual director returned no JSON object")
    try:
        value = json.loads(clean[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VisualPromptError(f"Visual director returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualPromptError("Visual director JSON must be an object")
    return value


def _selected_source_context(
    package: VideoPackage,
    sources: list[SourceItem],
) -> list[dict[str, Any]]:
    by_url = {source.url: source for source in sources}
    result: list[dict[str, Any]] = []
    for source_index, url in enumerate(package.source_urls):
        source = by_url.get(url)
        if source is None:
            raise VisualPromptError(f"Selected source is missing from research context: {url}")
        result.append(
            {
                "source_index": source_index,
                "publisher": source.publisher,
                "title": source.title,
                "summary": source.summary,
                "url": source.url,
            }
        )
    return result


def _director_input(
    package: VideoPackage,
    source_context: list[dict[str, Any]],
    strategy: Strategy,
) -> dict[str, Any]:
    return {
        "topic": package.topic,
        "title": package.title,
        "narration": package.narration,
        "strategy": {
            "hook": strategy.hook,
            "pacing": strategy.pacing,
            "visual": strategy.visual,
            "duration": strategy.duration,
        },
        "sources": source_context,
        "scenes": [
            {
                "scene_index": index,
                "heading": scene.heading,
                "body": scene.body,
                "visual_direction": scene.visual,
                "source_index": scene.source_index,
            }
            for index, scene in enumerate(package.scenes)
        ],
    }


def _director_prompt(payload: dict[str, Any], *, feedback: str = "") -> str:
    correction = (
        "\n\nTHE PREVIOUS VISUAL PLAN FAILED VALIDATION:\n"
        + feedback
        + "\nReturn a complete corrected JSON object."
        if feedback
        else ""
    )
    return f"""
You are the visual director for a factual vertical technology-news video. Construct a complete visual plan from the validated script and source context below. Return one JSON object only.

OBJECTIVE:
Create six coherent, premium, text-free scenes that illustrate the supplied claims without copying websites, logos, screenshots, trademarks, or real-person likenesses. The visual plan will drive an open image model for keyframes and Wan 2.2 for selected motion clips. Animated captions are rendered later and must never be baked into generated images or video.

GLOBAL RULES:
- Maintain one recognizable visual world across all six scenes: consistent materials, palette, lighting, lens language, and recurring abstract subject anchors.
- Every image_prompt must describe subject, environment, composition, camera, depth, material, lighting, and the factual idea represented.
- Keep the lower 32 percent visually quiet and free of essential subjects for animated captions.
- Never ask a model to render text, numbers, logos, watermarks, UI labels, article screenshots, branded products, or a named person's face.
- Use abstract or generic representations where the source discusses a company, product, model, policy, benchmark, or research result.
- Do not invent a visual fact. Use only the supplied scene claim and its selected source context.
- generation_mode must be "wan_i2v" for exactly three hero scenes and "image" for the other three.
- Hero scenes should be the opening hook and the two scenes with the strongest visual change or explanatory value.
- motion_prompt describes motion only: subject movement, environmental movement, parallax, lighting change, and camera movement. It must preserve the keyframe and introduce no new objects.
- Use controlled motion: no shake, whip pan, rapid orbit, morphing, object duplication, sudden cut, or unstable geometry.
- negative_prompt must explicitly reject text, logos, blur, deformation, flicker, and camera shake.
- scene_index and source_index must exactly match the supplied scene.

OUTPUT SHAPE:
{{
  "global_style": "...",
  "palette": "...",
  "lighting": "...",
  "continuity_bible": "...",
  "scenes": [
    {{
      "scene_index": 0,
      "source_index": 0,
      "role": "hook|evidence|mechanism|comparison|implication|cta",
      "generation_mode": "wan_i2v|image",
      "image_prompt": "45-180 words",
      "motion_prompt": "15-70 words",
      "negative_prompt": "comma-separated defects",
      "continuity_anchor": "recurring visual element"
    }}
  ]
}}

VALIDATED INPUT:
{json.dumps(payload, indent=2, ensure_ascii=False)}{correction}
""".strip()


def _chat_visual_director(settings: Settings, prompt: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    request = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise visual director. Return one JSON object only. "
                    "Never include markdown or hidden reasoning. /no_think"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": min(settings.llm_temperature, 0.55),
        "max_tokens": 4200,
        "stream": False,
        "response_format": {"type": "json_object", "schema": _VISUAL_PLAN_SCHEMA},
    }
    response = requests.post(
        f"{settings.llm_base_url}/chat/completions",
        headers=headers,
        json=request,
        timeout=settings.llm_timeout_seconds,
    )
    if response.status_code >= 400:
        raise VisualPromptError(
            f"Visual director model returned {response.status_code}: {response.text[:1200]}"
        )
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise VisualPromptError("Visual director response contained no choices")
    content = ((choices[0].get("message") or {}).get("content"))
    if not isinstance(content, str) or not content.strip():
        raise VisualPromptError("Visual director response contained no content")
    return _extract_json(content)


def _clean_prompt(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _stable_seed(title: str, scene_index: int) -> int:
    digest = hashlib.sha256(f"{title}|{scene_index}|{PROMPT_VERSION}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _planned_image_model() -> str:
    backend = os.getenv("VISUAL_IMAGE_BACKEND", "auto").strip().lower()
    if backend == "flux" or (backend == "auto" and os.getenv("HF_TOKEN")):
        return os.getenv(
            "VISUAL_FLUX_MODEL",
            "black-forest-labs/FLUX.1-schnell",
        ).strip()
    return (
        os.getenv("VISUAL_SDXL_LIGHTNING_REPO", "ByteDance/SDXL-Lightning").strip()
        + ":"
        + os.getenv(
            "VISUAL_SDXL_LIGHTNING_CHECKPOINT",
            "sdxl_lightning_4step_unet.safetensors",
        ).strip()
    )


def _validate_and_normalize(
    raw: dict[str, Any],
    *,
    package: VideoPackage,
    settings: Settings,
    director_input_sha256: str,
) -> VisualPlan:
    raw_scenes = raw.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != len(package.scenes):
        raise VisualPromptError("Visual plan must contain exactly one entry per package scene")

    by_index: dict[int, dict[str, Any]] = {}
    for value in raw_scenes:
        if not isinstance(value, dict):
            raise VisualPromptError("Every visual scene must be an object")
        try:
            index = int(value.get("scene_index"))
        except (TypeError, ValueError) as exc:
            raise VisualPromptError("Every visual scene requires an integer scene_index") from exc
        if index in by_index:
            raise VisualPromptError(f"Duplicate visual scene_index: {index}")
        by_index[index] = value

    expected_indices = set(range(len(package.scenes)))
    if set(by_index) != expected_indices:
        raise VisualPromptError("Visual scene indices must be contiguous and match the package")

    modes = {
        index: _clean_prompt(by_index[index].get("generation_mode")).lower()
        for index in sorted(by_index)
    }
    if sum(mode == "wan_i2v" for mode in modes.values()) != 3:
        raise VisualPromptError("Visual plan must route exactly three scenes to wan_i2v")
    if any(mode not in {"image", "wan_i2v"} for mode in modes.values()):
        raise VisualPromptError("Visual plan contains an unsupported generation_mode")
    if modes[0] != "wan_i2v":
        raise VisualPromptError("Opening hook scene must use wan_i2v")

    scene_prompts: list[SceneVisualPrompt] = []
    duration = max(3.0, min(6.0, settings.target_seconds / len(package.scenes)))
    for index, source_scene in enumerate(package.scenes):
        value = by_index[index]
        try:
            source_index = int(value.get("source_index"))
        except (TypeError, ValueError) as exc:
            raise VisualPromptError(f"Scene {index} source_index must be an integer") from exc
        if source_index != source_scene.source_index:
            raise VisualPromptError(
                f"Scene {index} source_index changed from {source_scene.source_index} to {source_index}"
            )

        role = _clean_prompt(value.get("role")).lower()
        if role not in {"hook", "evidence", "mechanism", "comparison", "implication", "cta"}:
            raise VisualPromptError(f"Scene {index} has invalid role: {role}")
        image_prompt = _clean_prompt(value.get("image_prompt"))
        motion_prompt = _clean_prompt(value.get("motion_prompt"))
        negative_prompt = _clean_prompt(value.get("negative_prompt"))
        continuity_anchor = _clean_prompt(value.get("continuity_anchor"))
        if not 35 <= len(image_prompt.split()) <= 220:
            raise VisualPromptError(f"Scene {index} image_prompt must contain 35-220 words")
        if not 12 <= len(motion_prompt.split()) <= 100:
            raise VisualPromptError(f"Scene {index} motion_prompt must contain 12-100 words")
        lowered = image_prompt.casefold()
        for directive in _FORBIDDEN_IMAGE_DIRECTIVES:
            if directive in lowered:
                raise VisualPromptError(
                    f"Scene {index} requests baked text or branding: {directive}"
                )
        if not continuity_anchor:
            raise VisualPromptError(f"Scene {index} continuity_anchor is empty")

        image_prompt = image_prompt.rstrip(" .") + "." + _IMAGE_SUFFIX
        motion_prompt = motion_prompt.rstrip(" .") + "." + _MOTION_SUFFIX
        negative_prompt = ", ".join(
            part for part in (negative_prompt, _DEFAULT_NEGATIVE) if part
        )
        scene_prompts.append(
            SceneVisualPrompt(
                scene_index=index,
                source_index=source_index,
                role=role,
                generation_mode=modes[index],
                image_prompt=image_prompt,
                motion_prompt=motion_prompt,
                negative_prompt=negative_prompt,
                continuity_anchor=continuity_anchor,
                caption_safe_zone="lower_32_percent",
                seed=_stable_seed(package.title, index),
                duration_seconds=round(duration, 3),
            )
        )

    global_style = _clean_prompt(raw.get("global_style"))
    palette = _clean_prompt(raw.get("palette"))
    lighting = _clean_prompt(raw.get("lighting"))
    continuity_bible = _clean_prompt(raw.get("continuity_bible"))
    if min(map(len, (global_style, palette, lighting, continuity_bible))) < 8:
        raise VisualPromptError("Global visual style fields are incomplete")

    return VisualPlan(
        prompt_version=PROMPT_VERSION,
        global_style=global_style,
        palette=palette,
        lighting=lighting,
        continuity_bible=continuity_bible,
        image_model=_planned_image_model(),
        video_model=os.getenv(
            "WAN22_MODEL_ID",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ).strip(),
        width=704,
        height=1280,
        fps=24,
        director_input_sha256=director_input_sha256,
        scenes=tuple(scene_prompts),
    )


def construct_visual_plan(
    settings: Settings,
    package: VideoPackage,
    sources: list[SourceItem],
    strategy: Strategy,
    *,
    plan_validator: Callable[[VisualPlan], None] | None = None,
) -> VisualPlan:
    """Construct and repair a plan until its executable production preflight passes."""
    if len(package.scenes) != 6:
        raise VisualPromptError("The visual director requires exactly six scenes")
    source_context = _selected_source_context(package, sources)
    payload = _director_input(package, source_context, strategy)
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload_sha = hashlib.sha256(payload_json.encode()).hexdigest()

    feedback = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = _chat_visual_director(
                settings,
                _director_prompt(payload, feedback=feedback),
            )
            plan = _validate_and_normalize(
                raw,
                package=package,
                settings=settings,
                director_input_sha256=payload_sha,
            )
            if plan_validator is not None:
                plan_validator(plan)
            return plan
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            if attempt + 1 < 3:
                time.sleep(2**attempt)

    assert last_error is not None
    raise VisualPromptError(
        f"Visual prompt construction failed after 3 attempts: {last_error}"
    ) from last_error
