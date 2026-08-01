from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .config import Settings
from .feeds import SourceItem
from .models import Scene, VideoPackage
from .policy import Strategy


class LocalLLMError(RuntimeError):
    pass


PACKAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skip_reason": {"type": "string"},
        "topic": {"type": "string"},
        "narration": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "thumbnail_text": {"type": "string"},
        "top_comment": {"type": "string"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
        "source_publishers": {"type": "array", "items": {"type": "string"}},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string"},
                    "visual": {"type": "string"},
                    "source_index": {"type": "integer", "minimum": 0},
                },
                "required": ["heading", "body", "visual", "source_index"],
            },
        },
    },
}


def _extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    if start < 0:
        raise LocalLLMError("Local model returned no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(clean)):
        char = clean[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = clean[start : index + 1]
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise LocalLLMError(f"Invalid JSON from local model: {exc}") from exc
                if not isinstance(data, dict):
                    raise LocalLLMError("Local model JSON must be an object")
                return data
    raise LocalLLMError("Local model returned an incomplete JSON object")


def _chat(settings: Settings, prompt: str, *, attempts: int = 3) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise research editor. Return one JSON object only. "
                    "Do not include markdown, analysis, or hidden reasoning. /no_think"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": 3000,
        "stream": False,
        "response_format": {"type": "json_object", "schema": PACKAGE_SCHEMA},
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                f"{settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
            if response.status_code >= 400:
                raise LocalLLMError(f"llama.cpp returned {response.status_code}: {response.text[:1500]}")
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LocalLLMError("llama.cpp response contained no choices")
            content = ((choices[0].get("message") or {}).get("content"))
            if not isinstance(content, str) or not content.strip():
                raise LocalLLMError("llama.cpp response contained no message content")
            return _extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise LocalLLMError(str(last_error)) from last_error


def healthcheck(settings: Settings) -> dict[str, Any]:
    response = requests.get(f"{settings.llm_base_url}/models", timeout=10)
    if response.status_code >= 400:
        raise LocalLLMError(f"llama.cpp health check failed: {response.status_code}")
    data = response.json()
    models = data.get("data") or []
    return {"ok": True, "models": [str(item.get("id", "")) for item in models]}


def _source_payload(sources: list[SourceItem]) -> list[dict[str, str]]:
    return [
        {
            "publisher": item.publisher,
            "title": item.title,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "summary": item.summary,
        }
        for item in sources[:30]
    ]


def _initial_prompt(
    settings: Settings,
    strategy: Strategy,
    source_payload: list[dict[str, str]],
) -> str:
    cta = (
        f"Use this owner-controlled CTA exactly once: {settings.monetization_label}: {settings.monetization_url}"
        if settings.monetization_url
        else "Use a natural subscribe CTA; do not invent a product, affiliate link, or revenue claim."
    )
    return f"""
Use ONLY the supplied primary-source feed entries. Select one current AI development supported by at least two supplied URLs. A second source may provide context rather than independent confirmation, but do not imply independent confirmation when it is not present. Never invent dates, benchmarks, pricing, availability, quotes, partnerships, or capabilities.

Creative strategy:
- hook: {strategy.hook}
- pacing: {strategy.pacing}
- visual style: {strategy.visual}
- duration target: {strategy.duration} seconds
- CTA type: {strategy.cta}
- {cta}

Return one JSON object containing:
- topic: string
- narration: 135-175 words, exact spoken script, credible and non-hyped. The final hard acceptance range is 120-190 words; count the words before returning.
- title: <=90 characters
- description: two concise paragraphs followed by a Sources section containing every used source URL
- tags: 8-14 plain strings. These are display metadata, not factual claims.
- thumbnail_text: 2-5 words and <=45 characters. This is display metadata.
- top_comment: one useful question plus CTA when configured. This is display metadata.
- source_urls: 2-5 URLs copied exactly from the supplied entries
- source_publishers: publisher names corresponding one-for-one with source_urls
- scenes: exactly 6 objects with heading <=5 words, body <=18 words, a procedural visual direction, and source_index. source_index is ZERO-BASED into the final source_urls array: the first source is 0 and the last valid value is len(source_urls)-1. Never use source_index equal to len(source_urls). Do not request copyrighted footage, logos, screenshots, or real-person likenesses.

When no topic satisfies the evidence and quality rules, return only {{"skip_reason":"specific reason"}}.

SOURCE ENTRIES:
{json.dumps(source_payload, ensure_ascii=False)}
""".strip()


def _repair_prompt(
    initial_prompt: str,
    previous: dict[str, Any],
    error: str,
) -> str:
    return f"""
{initial_prompt}

The previous JSON failed deterministic validation:
{error}

Return a COMPLETE corrected JSON object. Do not return a patch.
Preserve supplied source URLs exactly and do not add unsupported facts.
The narration must contain 135-175 words; the hard accepted range is 120-190 words. Count before returning.
source_index is ZERO-BASED into source_urls: valid values are 0 through len(source_urls)-1 only.

PREVIOUS JSON:
{json.dumps(previous, ensure_ascii=False)}
""".strip()


def _is_scene_source_index_error(error: Exception) -> bool:
    return str(error).startswith("Scene source_index")


def _scene_index_repair_prompt(
    previous: dict[str, Any],
    source_payload: list[dict[str, str]],
    error: str,
) -> str:
    source_urls = [str(value) for value in previous.get("source_urls") or []]
    if not source_urls:
        raise LocalLLMError("Cannot repair scene source indices without source_urls")
    available = {entry["url"]: entry for entry in source_payload}
    source_table = []
    for index, url in enumerate(source_urls):
        entry = available.get(url)
        if entry is None:
            raise LocalLLMError(f"Cannot repair unknown source URL: {url}")
        source_table.append(
            {
                "source_index": index,
                "publisher": entry["publisher"],
                "title": entry["title"],
                "url": entry["url"],
                "summary": entry["summary"],
            }
        )
    return f"""
The package is factually frozen and failed only this deterministic validation:
{error}

There are exactly {len(source_urls)} cited sources. Valid ZERO-BASED source_index values are 0 through {len(source_urls) - 1}. A value of {len(source_urls)} is invalid.

Return one COMPLETE JSON object. Preserve every field and every character from PREVIOUS JSON except the six scenes[*].source_index integers. Do not change topic, narration, title, description, tags, thumbnail_text, top_comment, source_urls, source_publishers, scene headings, scene bodies, or visual directions.

For each scene, choose the valid source_index whose supplied title or summary actually supports that scene's existing claim. Do not invent or reorder sources. Count six scenes before returning.

INDEXED SOURCE TABLE:
{json.dumps(source_table, ensure_ascii=False)}

PREVIOUS JSON:
{json.dumps(previous, ensure_ascii=False)}
""".strip()


def _merge_scene_indices(
    frozen: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    frozen_scenes = frozen.get("scenes")
    candidate_scenes = candidate.get("scenes")
    if not isinstance(frozen_scenes, list) or len(frozen_scenes) != 6:
        raise LocalLLMError("Cannot repair source indices without six frozen scenes")
    if not isinstance(candidate_scenes, list) or len(candidate_scenes) != 6:
        raise LocalLLMError("Scene-index repair must return exactly six scenes")

    merged = json.loads(json.dumps(frozen))
    merged_scenes: list[dict[str, Any]] = []
    for index, (frozen_scene, candidate_scene) in enumerate(
        zip(frozen_scenes, candidate_scenes, strict=True)
    ):
        if not isinstance(frozen_scene, dict) or not isinstance(candidate_scene, dict):
            raise LocalLLMError(f"Scene-index repair returned invalid scene {index}")
        if "source_index" not in candidate_scene:
            raise LocalLLMError(f"Scene-index repair omitted source_index for scene {index}")
        repaired_scene = dict(frozen_scene)
        repaired_scene["source_index"] = candidate_scene["source_index"]
        merged_scenes.append(repaired_scene)
    merged["scenes"] = merged_scenes
    return merged


def _word_count(text: str) -> int:
    return len(text.split())


def _fallback_tags(topic: str, strategy: Strategy) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.-]*", topic)
    stop = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    candidates = [word for word in words if word.casefold() not in stop and len(word) >= 2]
    candidates.extend(
        [
            "AI",
            "Artificial Intelligence",
            "AI News",
            "AI Engineering",
            "Machine Learning",
            strategy.hook,
            strategy.pacing,
            strategy.visual,
            "Technology",
            "Tech News",
        ]
    )
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = str(candidate).strip()[:40]
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
        if len(unique) == 14:
            break
    return unique


def _complete_tags(raw_tags: object, topic: str, strategy: Strategy) -> list[str]:
    supplied = raw_tags if isinstance(raw_tags, list) else []
    candidates = [str(tag).strip()[:40] for tag in supplied]
    candidates.extend(_fallback_tags(topic, strategy))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) == 14:
            break
    if len(unique) < 8:
        raise LocalLLMError("Unable to complete at least eight display tags")
    return unique


def _complete_thumbnail_text(raw_text: object, topic: str) -> str:
    supplied = str(raw_text or "").strip()
    supplied_words = supplied.split()
    if 2 <= len(supplied_words) <= 5 and len(supplied) <= 45:
        return supplied
    topic_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.-]*", topic)
    stop = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with"}
    meaningful = [word for word in topic_words if word.casefold() not in stop]
    selected = meaningful[:4]
    if len(selected) < 2:
        selected = ["AI", "UPDATE"]
    completed = " ".join(selected[:5]).upper()
    if len(completed) > 45:
        completed = " ".join(selected[:3]).upper()[:45].rstrip()
    if not 2 <= len(completed.split()) <= 5:
        completed = "AI UPDATE"
    return completed


def _complete_top_comment(raw_comment: object, settings: Settings) -> str:
    supplied = str(raw_comment or "").strip()
    if supplied:
        return supplied[:9000]
    if settings.monetization_url:
        return (
            "What would you test first? "
            f"{settings.monetization_label}: {settings.monetization_url}"
        )[:9000]
    return "What would you test first? Subscribe for evidence-backed AI updates."


def _validate_required_core(raw: dict[str, Any]) -> None:
    factual_required = {
        "topic",
        "narration",
        "title",
        "description",
        "source_urls",
        "source_publishers",
        "scenes",
    }
    missing = factual_required - set(raw)
    if missing:
        raise LocalLLMError("Package missing keys: " + ", ".join(sorted(missing)))


def _normalize_scene_source_indices(
    scenes_raw: list[dict[str, Any]],
    source_count: int,
) -> list[dict[str, Any]]:
    values: list[int] = []
    for scene in scenes_raw:
        try:
            values.append(int(scene["source_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LocalLLMError("Every scene requires an integer source_index") from exc
    if source_count > 0 and all(1 <= value <= source_count for value in values):
        return [dict(scene, source_index=value - 1) for scene, value in zip(scenes_raw, values, strict=True)]
    return scenes_raw


def _package_from_raw(
    raw: dict[str, Any],
    settings: Settings,
    sources: list[SourceItem],
    strategy: Strategy,
) -> VideoPackage:
    if raw.get("skip_reason"):
        raise LocalLLMError(f"No publishable trend: {raw['skip_reason']}")
    _validate_required_core(raw)
    source_urls = [str(value) for value in raw["source_urls"]]
    source_publishers = [str(value) for value in raw["source_publishers"]]
    allowed_by_url = {item.url: item.publisher for item in sources}
    if len(source_urls) < settings.min_primary_sources:
        raise LocalLLMError("Package did not cite enough supplied primary sources")
    if not set(source_urls).issubset(allowed_by_url):
        raise LocalLLMError("Package cited a URL that was not supplied")
    if len(source_publishers) != len(source_urls):
        raise LocalLLMError("source_publishers must correspond one-for-one with source_urls")
    for url, publisher in zip(source_urls, source_publishers, strict=True):
        if publisher.strip().casefold() != allowed_by_url[url].strip().casefold():
            raise LocalLLMError(f"Publisher mismatch for source URL: {url}")
    independent = {allowed_by_url[url].strip().casefold() for url in source_urls}
    if len(independent) < settings.min_primary_sources:
        raise LocalLLMError("Package did not use enough independent primary publishers")
    narration = str(raw["narration"]).strip()
    narration_words = _word_count(narration)
    if not 120 <= narration_words <= 190:
        raise LocalLLMError(
            "Narration word count outside quality gate: "
            f"{narration_words}; required 120-190 words"
        )
    scenes_raw = raw["scenes"]
    if not isinstance(scenes_raw, list) or len(scenes_raw) != 6:
        raise LocalLLMError("Exactly six scenes are required")
    scenes_raw = _normalize_scene_source_indices(scenes_raw, len(source_urls))
    scenes: list[Scene] = []
    for scene in scenes_raw:
        source_index = int(scene["source_index"])
        if not 0 <= source_index < len(source_urls):
            raise LocalLLMError(
                f"Scene source_index out of range: {source_index}; valid zero-based "
                f"source_urls range is 0-{len(source_urls) - 1}"
            )
        scenes.append(
            Scene(
                heading=str(scene["heading"])[:60],
                body=str(scene["body"])[:180],
                visual=str(scene["visual"])[:400],
                source_index=source_index,
            )
        )
    topic = str(raw["topic"]).strip()
    description = str(raw["description"]).strip()
    for source_url in source_urls:
        if source_url not in description:
            description += f"\n{source_url}"
    return VideoPackage(
        topic=topic,
        narration=narration,
        title=str(raw["title"]).strip()[:90],
        description=description[:4900],
        tags=_complete_tags(raw.get("tags"), topic, strategy),
        thumbnail_text=_complete_thumbnail_text(raw.get("thumbnail_text"), topic),
        top_comment=_complete_top_comment(raw.get("top_comment"), settings),
        scenes=scenes,
        source_urls=source_urls,
        source_publishers=source_publishers,
    )


def generate_package(settings: Settings, sources: list[SourceItem], strategy: Strategy) -> VideoPackage:
    source_payload = _source_payload(sources)
    initial_prompt = _initial_prompt(settings, strategy, source_payload)
    current_prompt = initial_prompt
    previous: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(3):
        raw = _chat(settings, current_prompt)
        previous = raw
        try:
            return _package_from_raw(raw, settings, sources, strategy)
        except LocalLLMError as exc:
            last_error = exc
            if _is_scene_source_index_error(exc):
                frozen = raw
                targeted_error: Exception = exc
                for _ in range(3):
                    candidate = _chat(
                        settings,
                        _scene_index_repair_prompt(
                            frozen,
                            source_payload,
                            str(targeted_error),
                        ),
                    )
                    try:
                        repaired = _merge_scene_indices(frozen, candidate)
                        return _package_from_raw(repaired, settings, sources, strategy)
                    except LocalLLMError as repair_exc:
                        targeted_error = repair_exc
                raise LocalLLMError(
                    "Scene source_index repair failed after 3 targeted attempts: "
                    f"{targeted_error}"
                ) from targeted_error
            if attempt + 1 < 3:
                current_prompt = _repair_prompt(initial_prompt, raw, str(exc))
    detail = str(last_error) if last_error else "unknown validation failure"
    if previous is None:
        raise LocalLLMError("Package generation returned no candidate")
    raise LocalLLMError(f"Package validation failed after 3 attempts: {detail}") from last_error
