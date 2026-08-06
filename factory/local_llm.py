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


PACKAGE_CORE_REQUIRED_KEYS = {
    "topic",
    "narration",
    "title",
    "description",
    "source_urls",
    "source_publishers",
    "scenes",
}

NARRATION_MIN_WORDS = 120
NARRATION_MAX_WORDS = 190
NARRATION_TARGET_MIN_WORDS = 145
NARRATION_TARGET_MAX_WORDS = 165

EVIDENCE_SAFE_CLOSES = (
    "Before adopting it, read the linked primary sources, test the feature on a controlled task, and compare the result with your current workflow.",
    "That separates an interesting release from a dependable production tool.",
    "The evidence should decide the next step, not the headline.",
)

DEFAULT_DISCOVERY_TAGS = (
    "AI",
    "Artificial Intelligence",
    "AI News",
    "AI Engineering",
    "Machine Learning",
    "AI Tools",
    "Technology",
    "AI Update",
)

THUMBNAIL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
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


def _word_count(text: str) -> int:
    return len(text.split())


def _stabilize_near_minimum_narration(raw: dict[str, Any]) -> dict[str, Any]:
    narration = str(raw.get("narration") or "").strip()
    word_count = _word_count(narration)
    if not 100 <= word_count < NARRATION_MIN_WORDS:
        return raw

    corrected = dict(raw)
    lowered = narration.casefold()
    for closing in EVIDENCE_SAFE_CLOSES:
        if closing.casefold() not in lowered:
            narration = f"{narration} {closing}".strip()
            lowered = narration.casefold()
        if _word_count(narration) >= 135:
            break
    corrected["narration"] = narration
    return corrected


def _deduplicated_nonempty_strings(values: Any, *, limit: int, max_length: int) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()[:max_length]
        key = text.casefold()
        if not text or key in seen:
            continue
        result.append(text)
        seen.add(key)
        if len(result) >= limit:
            break
    return result


def _topic_tag_candidates(raw: dict[str, Any]) -> list[str]:
    seed = f"{raw.get('topic', '')} {raw.get('title', '')}"
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{1,31}", seed)
    return [word for word in words if word.casefold() not in THUMBNAIL_STOP_WORDS]


def _derive_thumbnail_text(raw: dict[str, Any]) -> str:
    existing = str(raw.get("thumbnail_text") or "").strip()
    existing_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#'.-]*", existing)
    if 2 <= len(existing_words) <= 5:
        return " ".join(existing_words).upper()[:45]

    seed = str(raw.get("title") or raw.get("topic") or "AI Update")
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#'.-]*", seed)
    significant = [word for word in words if word.casefold() not in THUMBNAIL_STOP_WORDS]
    selected = (significant or words)[:5]
    if len(selected) < 2:
        selected = (selected + ["AI", "UPDATE"])[:2]
    return " ".join(selected).upper()[:45]


def _complete_display_metadata(raw: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Fill non-factual discovery metadata deterministically without altering claims."""
    corrected = dict(raw)

    tags = _deduplicated_nonempty_strings(corrected.get("tags"), limit=14, max_length=40)
    seen = {tag.casefold() for tag in tags}
    for candidate in (*_topic_tag_candidates(corrected), *DEFAULT_DISCOVERY_TAGS):
        text = candidate.strip()[:40]
        key = text.casefold()
        if not text or key in seen:
            continue
        tags.append(text)
        seen.add(key)
        if len(tags) >= 8:
            break
    corrected["tags"] = tags[:14]

    corrected["thumbnail_text"] = _derive_thumbnail_text(corrected)

    top_comment = str(corrected.get("top_comment") or "").strip()
    if not top_comment:
        top_comment = "What would you test first?"
        if settings.monetization_url:
            label = settings.monetization_label.strip() or "Learn more"
            top_comment += f" {label}: {settings.monetization_url}"
        else:
            top_comment += " Subscribe for evidence-backed AI updates."
    corrected["top_comment"] = top_comment
    return corrected


def _balanced_source_candidates(
    sources: list[SourceItem],
    *,
    limit: int = 30,
) -> list[SourceItem]:
    """Round-robin recent sources by publisher so one feed cannot dominate the prompt."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    grouped: dict[str, list[SourceItem]] = {}
    publisher_order: list[str] = []
    for source in sources:
        key = source.publisher.strip().casefold()
        if key not in grouped:
            grouped[key] = []
            publisher_order.append(key)
        grouped[key].append(source)

    selected: list[SourceItem] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in publisher_order:
            group = grouped[key]
            if depth >= len(group):
                continue
            selected.append(group[depth])
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        depth += 1
    return selected


def _publisher_source_table(
    source_payload: list[dict[str, Any]],
    *,
    urls_per_publisher: int = 4,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in source_payload:
        publisher = str(item.get("publisher") or "").strip()
        key = publisher.casefold()
        if key not in grouped:
            grouped[key] = {"publisher": publisher, "sources": []}
            order.append(key)
        grouped_sources = grouped[key]["sources"]
        if len(grouped_sources) < urls_per_publisher:
            grouped_sources.append(
                {
                    "source_id": item.get("source_id"),
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                }
            )
    return [grouped[key] for key in order]


def _selected_publishers(
    previous: dict[str, Any],
    source_payload: list[dict[str, Any]],
) -> list[str]:
    selected = previous.get("source_urls")
    if not isinstance(selected, list):
        return []
    publisher_by_url = {
        str(item.get("url") or ""): str(item.get("publisher") or "").strip()
        for item in source_payload
    }
    publishers: list[str] = []
    seen: set[str] = set()
    for raw_url in selected:
        publisher = publisher_by_url.get(str(raw_url), "")
        key = publisher.casefold()
        if publisher and key not in seen:
            publishers.append(publisher)
            seen.add(key)
    return publishers


def _normalize_scene_source_indices(
    scenes_raw: list[Any],
    source_urls: list[str],
    sources: list[SourceItem],
) -> list[int]:
    raw_indices: list[int] = []
    for scene in scenes_raw:
        if not isinstance(scene, dict) or "source_index" not in scene:
            raise LocalLLMError("Every scene requires an integer source_index")
        value = scene["source_index"]
        if isinstance(value, bool):
            raise LocalLLMError("Every scene requires an integer source_index")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise LocalLLMError("Every scene requires an integer source_index") from exc
        raw_indices.append(index)

    if all(0 <= index < len(source_urls) for index in raw_indices):
        return raw_indices

    selected_index_by_url = {url: index for index, url in enumerate(source_urls)}
    candidates: dict[str, tuple[int, ...]] = {}

    if all(1 <= index <= len(source_urls) for index in raw_indices):
        candidates["one-based source_urls"] = tuple(index - 1 for index in raw_indices)

    for label, offset in (("zero-based SOURCE ENTRIES", 0), ("one-based SOURCE ENTRIES", 1)):
        mapped: list[int] = []
        for index in raw_indices:
            source_position = index - offset
            if not 0 <= source_position < len(sources):
                mapped = []
                break
            selected_index = selected_index_by_url.get(sources[source_position].url)
            if selected_index is None:
                mapped = []
                break
            mapped.append(selected_index)
        if mapped:
            candidates[label] = tuple(mapped)

    unique_mappings: dict[tuple[int, ...], list[str]] = {}
    for label, mapping in candidates.items():
        unique_mappings.setdefault(mapping, []).append(label)

    if len(unique_mappings) == 1:
        return list(next(iter(unique_mappings)))
    if len(unique_mappings) > 1:
        conventions = "; ".join(
            f"{', '.join(labels)} -> {list(mapping)}"
            for mapping, labels in unique_mappings.items()
        )
        raise LocalLLMError(
            "Scene source_index convention is ambiguous; use zero-based positions "
            f"from source_urls only. Candidates: {conventions}"
        )

    invalid = next(
        (index for index in raw_indices if not 0 <= index < len(source_urls)),
        raw_indices[0],
    )
    raise LocalLLMError(
        f"Scene source_index out of range: {invalid}; valid zero-based source_urls "
        f"range is 0-{len(source_urls) - 1}"
    )


def _package_from_raw(
    settings: Settings,
    sources: list[SourceItem],
    raw: dict[str, Any],
) -> VideoPackage:
    if raw.get("skip_reason"):
        raise LocalLLMError(f"No publishable trend: {raw['skip_reason']}")

    missing = PACKAGE_CORE_REQUIRED_KEYS - set(raw)
    if missing:
        raise LocalLLMError("Package missing keys: " + ", ".join(sorted(missing)))

    raw = _complete_display_metadata(raw, settings)

    source_urls = [str(value) for value in raw["source_urls"]]
    source_publishers = [str(value) for value in raw["source_publishers"]]
    allowed_by_url = {item.url: item.publisher for item in sources}
    if len(source_urls) < settings.min_primary_sources:
        raise LocalLLMError("Package did not cite enough supplied primary sources")
    if len(set(source_urls)) != len(source_urls):
        raise LocalLLMError("source_urls must not contain duplicates")
    if not set(source_urls).issubset(allowed_by_url):
        raise LocalLLMError("Package cited a URL that was not supplied")
    if len(source_publishers) != len(source_urls):
        raise LocalLLMError("source_publishers must correspond one-for-one with source_urls")
    for url, publisher in zip(source_urls, source_publishers, strict=True):
        if publisher.strip().casefold() != allowed_by_url[url].strip().casefold():
            raise LocalLLMError(f"Publisher mismatch for source URL: {url}")
    independent = {allowed_by_url[url].strip().casefold() for url in source_urls}
    if len(independent) < settings.min_primary_sources:
        raise LocalLLMError(
            f"Package used {len(independent)} independent primary publisher(s); "
            f"required {settings.min_primary_sources}"
        )

    narration = str(raw["narration"]).strip()
    word_count = _word_count(narration)
    if not NARRATION_MIN_WORDS <= word_count <= NARRATION_MAX_WORDS:
        raise LocalLLMError(f"Narration word count outside quality gate: {word_count}")

    scenes_raw = raw["scenes"]
    if not isinstance(scenes_raw, list) or len(scenes_raw) != 6:
        raise LocalLLMError("Exactly six scenes are required")
    source_indices = _normalize_scene_source_indices(scenes_raw, source_urls, sources)
    scenes: list[Scene] = []
    for scene, source_index in zip(scenes_raw, source_indices, strict=True):
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


def _selected_source_index_table(
    previous: dict[str, Any],
    source_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = previous.get("source_urls")
    if not isinstance(selected, list):
        return []
    catalog = {str(item.get("url")): item for item in source_payload}
    table: list[dict[str, Any]] = []
    for source_index, raw_url in enumerate(selected):
        url = str(raw_url)
        item = catalog.get(url, {})
        table.append(
            {
                "source_index": source_index,
                "url": url,
                "publisher": str(item.get("publisher") or ""),
            }
        )
    return table


def _repair_prompt(
    *,
    validation_error: str,
    previous: dict[str, Any],
    source_payload: list[dict[str, Any]],
    min_primary_sources: int,
) -> str:
    current_word_count = _word_count(str(previous.get("narration") or ""))
    minimum_addition = max(0, NARRATION_TARGET_MIN_WORDS - current_word_count)
    source_index_table = _selected_source_index_table(previous, source_payload)
    publisher_table = _publisher_source_table(source_payload)
    current_publishers = _selected_publishers(previous, source_payload)
    valid_max = len(source_index_table) - 1
    return f"""
Repair the previous JSON package so it passes the stated validation error. Return the COMPLETE corrected JSON object only, not a patch.

VALIDATION ERROR:
{validation_error}

PUBLISHER-DIVERSITY REPAIR:
- The package must cite at least {min_primary_sources} DISTINCT publisher names.
- The currently selected distinct publishers are: {json.dumps(current_publishers, ensure_ascii=False)}.
- Multiple URLs from one publisher still count as one publisher.
- Choose source_urls from at least {min_primary_sources} different rows in PUBLISHER SOURCE OPTIONS.
- Copy each publisher name exactly into the matching position of source_publishers.
- When replacing a source, rewrite any narration, scene, title, or description claim that the replacement does not support.
- Never substitute a URL while retaining an unsupported claim, and never invent cross-source confirmation.

PUBLISHER SOURCE OPTIONS:
{json.dumps(publisher_table, ensure_ascii=False)}

NARRATION REPAIR:
- The previous narration contains {current_word_count} whitespace-separated words.
- Rewrite the complete narration to {NARRATION_TARGET_MIN_WORDS}-{NARRATION_TARGET_MAX_WORDS} words.
- Add at least {minimum_addition} words when the previous narration is too short.
- Do not merely append a fragment; preserve a natural hook, evidence, practical implication, caveat, and closing.
- Count the final narration words before returning the JSON.

SCENE SOURCE-INDEX REPAIR:
- source_index is a zero-based position in the JSON package's own source_urls array.
- source_index is NOT a SOURCE ENTRIES source_id and must never copy that catalog identifier.
- The only valid source_index values for this package are 0 through {valid_max}.
- Replace every scene source_index using the exact VALID SCENE SOURCE INDEX TABLE below after finalizing source_urls.
- Never clamp, wrap, or point a scene at a source that does not support its claim.

VALID SCENE SOURCE INDEX TABLE FROM THE PREVIOUS PACKAGE:
{json.dumps(source_index_table, ensure_ascii=False)}

NON-NEGOTIABLE RULES:
- Use only source URLs and publisher names copied exactly from SOURCE ENTRIES.
- source_urls must be unique.
- Preserve factual claims only when the selected supplied sources support them.
- scenes must contain exactly 6 objects.
- Every scene source_index must be valid for the corrected source_urls array.
- source_urls and source_publishers must correspond one-for-one.
- Keep title <=90 characters, thumbnail_text 2-5 words, and tags between 8 and 14 items.
- Do not return skip_reason unless the sources truly cannot support a publishable package.

PREVIOUS JSON:
{json.dumps(previous, ensure_ascii=False)}

SOURCE ENTRIES:
{json.dumps(source_payload, ensure_ascii=False)}
""".strip()


def generate_package(settings: Settings, sources: list[SourceItem], strategy: Strategy) -> VideoPackage:
    source_candidates = _balanced_source_candidates(sources, limit=30)
    source_payload = [
        {
            "source_id": source_id,
            "publisher": item.publisher,
            "author": item.author,
            "authority": item.authority,
            "title": item.title,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "summary": item.summary,
        }
        for source_id, item in enumerate(source_candidates)
    ]
    publisher_table = _publisher_source_table(source_payload)
    cta = (
        f"Use this owner-controlled CTA exactly once: {settings.monetization_label}: {settings.monetization_url}"
        if settings.monetization_url
        else "Use a natural subscribe CTA; do not invent a product, affiliate link, or revenue claim."
    )
    prompt = f"""
Use ONLY the supplied primary-source feed entries. Select one current AI development that can be responsibly explained using at least {settings.min_primary_sources} DISTINCT supplied publishers. A second publisher may provide context rather than independent confirmation, but do not imply independent confirmation when it is not present. Never invent dates, benchmarks, pricing, availability, quotes, partnerships, or capabilities.

AUTHORSHIP CONTRACT:
- ``publisher`` identifies the distribution feed or hosting platform; it is not automatically the company that created, launched, or released the subject.
- ``authority`` identifies the article author or organization responsible for the announcement.
- Attribute actions such as launched, released, introduced, published, or announced to ``authority`` only when the supplied title or summary supports that action.
- A hosting platform may be named as the place where a model is available, but never as its creator merely because it hosts the article or model page.

Creative strategy:
- hook: {strategy.hook}
- pacing: {strategy.pacing}
- visual style: {strategy.visual}
- duration target: {strategy.duration} seconds
- CTA type: {strategy.cta}
- {cta}

Return one JSON object containing:
- topic: string
- narration: {NARRATION_TARGET_MIN_WORDS}-{NARRATION_TARGET_MAX_WORDS} whitespace-separated words, exact spoken script, credible and non-hyped. Count the words before returning.
- title: <=90 characters
- description: two concise paragraphs followed by a Sources section containing every used source URL
- tags: 8-14 plain strings
- thumbnail_text: 2-5 words
- top_comment: one useful question plus CTA when configured
- source_urls: 2-5 UNIQUE URLs copied exactly from the supplied entries and spanning at least {settings.min_primary_sources} distinct publishers
- source_publishers: publisher names corresponding one-for-one with source_urls
- scenes: exactly 6 objects with heading <=5 words, body <=18 words, a procedural visual direction, and source_index. Do not request copyrighted footage, logos, screenshots, or real-person likenesses.

PUBLISHER-DIVERSITY CONTRACT:
- Choose source_urls from at least {settings.min_primary_sources} different rows in PUBLISHER SOURCE OPTIONS.
- Multiple URLs from one publisher still count as one publisher.
- Copy publisher names exactly and keep source_publishers aligned one-for-one with source_urls.
- If the sources cannot support one coherent package across that many publishers, return skip_reason rather than weakening attribution.

PUBLISHER SOURCE OPTIONS:
{json.dumps(publisher_table, ensure_ascii=False)}

SOURCE-INDEX CONTRACT:
- SOURCE ENTRIES source_id values identify catalog rows only.
- scene.source_index must NOT copy source_id.
- scene.source_index is the zero-based position in your returned source_urls array.
- Valid scene.source_index values are therefore 0 through len(source_urls)-1.
- Example: when source_urls is ["https://b.example", "https://a.example"], the first URL uses source_index 0 and the second uses source_index 1, regardless of their SOURCE ENTRIES source_id values.

When no topic satisfies the evidence and quality rules, return only {{"skip_reason":"specific reason"}}.

SOURCE ENTRIES:
{json.dumps(source_payload, ensure_ascii=False)}
""".strip()

    current_prompt = prompt
    last_error: LocalLLMError | None = None
    for package_attempt in range(3):
        raw = _chat(settings, current_prompt)
        if package_attempt == 2:
            raw = _stabilize_near_minimum_narration(raw)
        try:
            return _package_from_raw(settings, source_candidates, raw)
        except LocalLLMError as exc:
            if raw.get("skip_reason"):
                raise
            last_error = exc
            if package_attempt == 2:
                break
            current_prompt = _repair_prompt(
                validation_error=str(exc),
                previous=raw,
                source_payload=source_payload,
                min_primary_sources=settings.min_primary_sources,
            )

    assert last_error is not None
    raise LocalLLMError(f"Package validation failed after 3 attempts: {last_error}") from last_error
