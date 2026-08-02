from __future__ import annotations

import contextvars
import re
from collections import Counter
from dataclasses import replace
from typing import Any

from .feeds import SourceItem
from .models import VideoPackage


_INSTALLED = False
_FEEDBACK: contextvars.ContextVar[str] = contextvars.ContextVar(
    "production_content_feedback",
    default="",
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.-]{2,}")
_GENERIC = {
    "about",
    "across",
    "advancement",
    "advancements",
    "artificial",
    "development",
    "developments",
    "future",
    "innovation",
    "intelligence",
    "research",
    "technology",
    "workforce",
}
_BANNED_PHRASES = (
    "ai advancements",
    "advancements in ai",
    "future of ai",
    "shaping the future",
    "leading advancements",
    "more effective, ethical, and accessible",
    "scientific research and workforce readiness",
)
_MALFORMED_COPY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brechanging\b", "corrupted word 'rechanging'"),
    (r"\bis used of\b", "corrupted phrase 'is used of'"),
    (r"\bthe\s+the\b", "duplicated article"),
    (r"\bof\s+of\b", "duplicated preposition"),
    (r"\bhow\s+[^.!?]{0,120}\s+is\s+used\s+of\b", "malformed replacement clause"),
)
_RELATIONSHIP_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcollaboration\s+between\b", "collaboration between"),
    (r"\bpartnership\s+between\b", "partnership between"),
    (r"\bjointly\s+(?:developed|released|built|created)\b", "joint development"),
    (r"\bco-developed\b", "co-development"),
    (r"\bworked\s+together\b", "worked together"),
    (r"\bin\s+partnership\s+with\b", "partnership"),
)
_RELATIONSHIP_EVIDENCE_TERMS = (
    "collaboration",
    "collaborat",
    "partnership",
    "partnered",
    "jointly",
    "co-developed",
    "worked together",
    "in partnership with",
)
_RULES = """
PRODUCTION EDITORIAL RULES:
- Tell ONE coherent, current story. One source may be the main evidence and another may add directly relevant context; never splice unrelated announcements into a broad trend.
- The title and opening sentence must name a concrete company, product, policy, model, benchmark, or release from the supplied source titles.
- Open with the specific change and its consequence in the first 12 words. Do not begin with a generic industry overview.
- Narration must be 130-155 words so the finished vertical video lands near 55-62 seconds at short-form pace.
- Avoid generic filler such as “AI advancements,” “leading advancements,” “shaping the future,” and “more effective, ethical, and accessible.”
- Never describe selected publishers as collaborating, partnering, jointly developing, or confirming one another unless a supplied source explicitly states that relationship.
- Every scene must add a distinct fact or implication. Do not repeat the same conclusion in different wording.
- Thumbnail text must name the concrete subject in 2-5 words.
- Return zero-based source_index values only: with N source_urls, valid values are 0 through N-1.
""".strip()


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value).casefold())
        if token not in _GENERIC
    }


def _selected_sources(
    package: VideoPackage,
    sources: list[SourceItem],
) -> list[SourceItem]:
    by_url = {source.url: source for source in sources}
    return [by_url[url] for url in package.source_urls if url in by_url]


def _short_subject(source: SourceItem) -> str:
    clean = " ".join(source.title.replace("|", " ").replace(":", " ").split())
    words = clean.split()
    subject = " ".join(words[:7]).strip(" -—:;,.!")
    if not subject:
        subject = source.publisher.strip()
    return subject[:64].rstrip(" -—:;,.")


def _replace_phrase(text: str, phrase: str, replacement: str) -> str:
    """Replace a complete phrase only, never a substring inside another word."""
    pattern = rf"(?<![\w-]){re.escape(phrase)}(?![\w-])"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _ground_generic_copy(
    package: VideoPackage,
    sources: list[SourceItem],
) -> VideoPackage:
    """Replace generic slogans with grammatical, source-grounded language."""
    selected = _selected_sources(package, sources)
    if not selected:
        return package
    primary = selected[0]
    subject = _short_subject(primary)
    publisher = primary.publisher.strip()
    replacements: tuple[tuple[str, str], ...] = (
        ("reshaping the future of work", "changing how people work"),
        ("shaping the future of work", "changing how people work"),
        ("scientific research and workforce readiness", subject),
        ("more effective, ethical, and accessible", "more measurable and easier to evaluate"),
        ("leading advancements", f"new work on {subject}"),
        ("advancements in ai", subject),
        ("ai advancements", subject),
        ("future of ai", f"practical impact of {subject}"),
        ("shaping the future", "changing current practice"),
    )

    def edit(text: str) -> str:
        edited = text
        for phrase, replacement in replacements:
            edited = _replace_phrase(edited, phrase, replacement)
        return " ".join(edited.split())

    title = edit(package.title)
    source_specific = _tokens(primary.title) | _tokens(primary.publisher)
    if not (_tokens(title) & source_specific):
        title = f"{publisher}: {subject}"
    if len(title) < 28:
        title = f"{title} — What Changed"
    title = title[:78].rstrip(" -—:;,.")

    narration = edit(package.narration)
    sentences = re.split(r"(?<=[.!?])\s+", narration.strip(), maxsplit=1)
    if sentences and not (_tokens(sentences[0]) & source_specific):
        concrete_hook = f"{publisher} just detailed {subject}, and the change matters now."
        narration = " ".join([concrete_hook, *sentences[1:]])

    thumbnail = edit(package.thumbnail_text)
    if not (_tokens(thumbnail) & source_specific):
        thumbnail = " ".join(subject.split()[:5])
    thumbnail = thumbnail[:45].rstrip(" -—:;,.")

    scenes = [
        replace(
            scene,
            heading=edit(scene.heading)[:60],
            body=edit(scene.body)[:180],
            visual=edit(scene.visual)[:400],
        )
        for scene in package.scenes
    ]
    return replace(
        package,
        topic=edit(package.topic),
        narration=narration,
        title=title,
        description=edit(package.description),
        thumbnail_text=thumbnail,
        top_comment=edit(package.top_comment),
        scenes=scenes,
    )


def _validate_copy_integrity(package: VideoPackage) -> None:
    from .local_llm import LocalLLMError

    combined = " ".join(
        (
            package.topic,
            package.title,
            package.narration,
            package.description,
            package.thumbnail_text,
            package.top_comment,
            *(scene.heading for scene in package.scenes),
            *(scene.body for scene in package.scenes),
        )
    )
    for pattern, description in _MALFORMED_COPY_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            raise LocalLLMError(f"Production copy contains {description}")


def _validate_source_relationships(
    package: VideoPackage,
    selected: list[SourceItem],
) -> None:
    """Reject invented relationships between otherwise independent publishers."""
    from .local_llm import LocalLLMError

    copy = f"{package.title} {package.narration}".casefold()
    evidence = " ".join(
        f"{source.publisher} {source.title} {source.summary}" for source in selected
    ).casefold()
    evidence_declares_relationship = any(
        term in evidence for term in _RELATIONSHIP_EVIDENCE_TERMS
    )
    if evidence_declares_relationship:
        return
    for pattern, label in _RELATIONSHIP_PATTERNS:
        if re.search(pattern, copy, flags=re.IGNORECASE):
            raise LocalLLMError(
                "Production narration invents an unsupported cross-source relationship: "
                + label
            )


def _validate_publishable_content(
    package: VideoPackage,
    sources: list[SourceItem],
) -> None:
    from .local_llm import LocalLLMError

    _validate_copy_integrity(package)
    title_lower = package.title.casefold()
    narration_lower = package.narration.casefold()
    for phrase in _BANNED_PHRASES:
        if phrase in title_lower or phrase in narration_lower:
            raise LocalLLMError(f"Production copy contains generic phrase: {phrase}")

    word_count = len(package.narration.split())
    if not 130 <= word_count <= 155:
        raise LocalLLMError(
            f"Production narration must contain 130-155 words; received {word_count}"
        )
    if not 28 <= len(package.title) <= 78:
        raise LocalLLMError(
            f"Production title must contain 28-78 characters; received {len(package.title)}"
        )

    selected = _selected_sources(package, sources)
    source_specific: set[str] = set()
    for source in selected:
        source_specific.update(_tokens(source.title))
        source_specific.update(_tokens(source.publisher))
    if not source_specific:
        raise LocalLLMError("Selected sources contain no concrete title or publisher terms")

    _validate_source_relationships(package, selected)

    first_sentence = re.split(r"(?<=[.!?])\s+", package.narration.strip(), maxsplit=1)[0]
    if not (_tokens(package.title) & source_specific):
        raise LocalLLMError(
            "Production title does not name a concrete supplied source subject"
        )
    if not (_tokens(first_sentence) & source_specific):
        raise LocalLLMError(
            "Opening sentence does not name a concrete supplied source subject"
        )

    bodies = [" ".join(scene.body.casefold().split()) for scene in package.scenes]
    duplicates = [body for body, count in Counter(bodies).items() if count > 1]
    if duplicates:
        raise LocalLLMError("Production scenes repeat the same body text")

    sentence_starts = []
    for sentence in re.split(r"(?<=[.!?])\s+", package.narration):
        words = sentence.casefold().split()
        if len(words) >= 5:
            sentence_starts.append(" ".join(words[:5]))
    repeated_starts = [value for value, count in Counter(sentence_starts).items() if count > 1]
    if repeated_starts:
        raise LocalLLMError("Production narration repeats sentence openings")


def install_production_content_gate() -> None:
    """Strengthen prompts, ground generic copy, and retry non-publishable packages."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import local_llm

    original_chat = local_llm._chat
    original_generate = local_llm.generate_package

    def production_chat(
        settings: Any,
        prompt: str,
        *,
        attempts: int = 3,
    ) -> dict[str, Any]:
        feedback = _FEEDBACK.get()
        strengthened = prompt + "\n\n" + _RULES
        if feedback:
            strengthened += (
                "\n\nTHE PREVIOUS PACKAGE FAILED THIS PRODUCTION REVIEW:\n"
                + feedback
                + "\nReturn a complete corrected JSON package."
            )
        return original_chat(settings, strengthened, attempts=attempts)

    def production_generate(
        settings: Any,
        sources: list[SourceItem],
        strategy: Any,
    ) -> VideoPackage:
        from .local_llm import LocalLLMError

        last_error: Exception | None = None
        feedback = ""
        for _attempt in range(3):
            token = _FEEDBACK.set(feedback)
            try:
                package = original_generate(settings, sources, strategy)
            finally:
                _FEEDBACK.reset(token)
            package = _ground_generic_copy(package, sources)
            try:
                _validate_publishable_content(package, sources)
                return package
            except LocalLLMError as exc:
                last_error = exc
                feedback = str(exc)
        assert last_error is not None
        raise LocalLLMError(
            f"Production editorial review failed after 3 packages: {last_error}"
        ) from last_error

    local_llm._chat = production_chat
    local_llm.generate_package = production_generate
    _INSTALLED = True
