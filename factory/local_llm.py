from __future__ import annotations

import json
import re

import requests

from .config import Settings
from .feeds import SourceItem
from .models import Scene, VideoPackage
from .policy import Strategy


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("local model returned no JSON")
    return json.loads(text[start:end + 1])


def healthcheck(settings: Settings) -> dict:
    response = requests.get(f"{settings.llm_base_url}/models", timeout=10)
    response.raise_for_status()
    return {"ok": True, "models": [item.get("id", "") for item in response.json().get("data", [])]}


def generate_package(settings: Settings, sources: list[SourceItem], strategy: Strategy) -> VideoPackage:
    prompt = {
        "instruction": "Create one evidence-bound AI news short. Return JSON only. Never invent facts.",
        "strategy": strategy.key,
        "requirements": {"narration_words": "135-175", "scenes": 6, "sources": "2-5 supplied URLs"},
        "sources": [item.__dict__ | {"published_at": item.published_at.isoformat()} for item in sources[:30]],
    }
    response = requests.post(f"{settings.llm_base_url}/chat/completions", json={
        "model": settings.llm_model,
        "messages": [{"role": "system", "content": "Return one JSON object only. /no_think"}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        "temperature": settings.llm_temperature,
        "max_tokens": 3000,
        "stream": False,
    }, timeout=settings.llm_timeout_seconds)
    response.raise_for_status()
    raw = _extract_json(response.json()["choices"][0]["message"]["content"])
    allowed = {item.url: item.publisher for item in sources}
    urls = [str(value) for value in raw.get("source_urls", [])]
    publishers = [str(value) for value in raw.get("source_publishers", [])]
    if len(urls) < settings.min_primary_sources or not set(urls).issubset(allowed):
        raise RuntimeError("generated package failed source validation")
    if len(urls) != len(publishers) or any(allowed[url].casefold() != publisher.casefold() for url, publisher in zip(urls, publishers, strict=True)):
        raise RuntimeError("generated package has incorrect publisher mapping")
    scenes = [Scene(str(item["heading"])[:60], str(item["body"])[:180], str(item["visual"])[:400], int(item["source_index"])) for item in raw.get("scenes", [])]
    if len(scenes) != 6 or any(not 0 <= scene.source_index < len(urls) for scene in scenes):
        raise RuntimeError("generated package must contain six valid scenes")
    narration = str(raw.get("narration", "")).strip()
    if not 120 <= len(narration.split()) <= 190:
        raise RuntimeError("narration length failed quality gate")
    description = str(raw.get("description", "")).strip() + "\n\nSources:\n" + "\n".join(urls)
    return VideoPackage(str(raw.get("topic", "AI update")), narration, str(raw.get("title", "AI Update"))[:90], description[:4900], [str(tag)[:40] for tag in raw.get("tags", [])][:14], str(raw.get("thumbnail_text", "AI UPDATE"))[:45], str(raw.get("top_comment", ""))[:9000], scenes, urls, publishers)
