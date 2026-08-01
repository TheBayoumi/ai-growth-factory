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


def generate_package(settings: Settings, sources: list[SourceItem], strategy: Strategy) -> VideoPackage:
    source_payload = [
        {
            "publisher": item.publisher,
            "title": item.title,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "summary": item.summary,
        }
        for item in sources[:30]
    ]
    cta = (
        f"Use this owner-controlled CTA exactly once: {settings.monetization_label}: {settings.monetization_url}"
        if settings.monetization_url
        else "Use a natural subscribe CTA; do not invent a product, affiliate link, or revenue claim."
    )
    prompt = f"""
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
- narration: 135-175 words, exact spoken script, credible and non-hyped
- title: <=90 characters
- description: two concise paragraphs followed by a Sources section containing every used source URL
- tags: 8-14 plain strings
- thumbnail_text: 2-5 words
- top_comment: one useful question plus CTA when configured
- source_urls: 2-5 URLs copied exactly from the supplied entries
- source_publishers: publisher names corresponding one-for-one with source_urls
- scenes: exactly 6 objects with heading <=5 words, body <=18 words, a procedural visual direction, and source_index. source_index must identify the exact source_urls entry supporting that scene. Do not request copyrighted footage, logos, screenshots, or real-person likenesses.

When no topic satisfies the evidence and quality rules, return only {{"skip_reason":"specific reason"}}.

SOURCE ENTRIES:
{json.dumps(source_payload, ensure_ascii=False)}
""".strip()
    raw = _chat(settings, prompt)
    if raw.get("skip_reason"):
        raise LocalLLMError(f"No publishable trend: {raw['skip_reason']}")
    required = {
        "topic",
        "narration",
        "title",
        "description",
        "tags",
        "thumbnail_text",
        "top_comment",
        "source_urls",
        "source_publishers",
        "scenes",
    }
    missing = required - set(raw)
    if missing:
        raise LocalLLMError("Package missing keys: " + ", ".join(sorted(missing)))
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
    word_count = len(narration.split())
    if not 120 <= word_count <= 190:
        raise LocalLLMError(f"Narration word count outside quality gate: {word_count}")
    scenes_raw = raw["scenes"]
    if not isinstance(scenes_raw, list) or len(scenes_raw) != 6:
        raise LocalLLMError("Exactly six scenes are required")
    scenes: list[Scene] = []
    for scene in scenes_raw:
        source_index = int(scene["source_index"])
        if not 0 <= source_index < len(source_urls):
            raise LocalLLMError(f"Scene source_index out of range: {source_index}")
        scenes.append(
            Scene(
                heading=str(scene["heading"])[:60],
                body=str(scene["body"])[:180],
                visual=str(scene["visual"])[:400],
                source_index=source_index,
            )
        )
    description = str(raw["description"]).strip()
    for source_url in source_urls:
        if source_url not in description:
            description += f"\n{source_url}"
    return VideoPackage(
        topic=str(raw["topic"]).strip(),
        narration=narration,
        title=str(raw["title"]).strip()[:90],
        description=description[:4900],
        tags=[str(tag).strip()[:40] for tag in raw["tags"]][:14],
        thumbnail_text=str(raw["thumbnail_text"]).strip()[:45],
        top_comment=str(raw["top_comment"]).strip()[:9000],
        scenes=scenes,
        source_urls=source_urls,
        source_publishers=source_publishers,
    )
