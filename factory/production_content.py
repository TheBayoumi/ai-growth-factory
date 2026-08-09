from __future__ import annotations

import contextvars
import re
from collections import Counter
from dataclasses import replace
from typing import Any

from .feeds import SourceItem, source_authority
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
    "marks a significant step in making",
    "provides developers with a powerful tool",
    "making it easier to integrate ai into existing workflows",
    "a practical solution for developers and businesses",
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
    (r"\bstrategic\s+partnerships?\b", "strategic partnership"),
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
_RELEASE_ACTOR_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9&.+-]*(?:\s+[A-Z][A-Za-z0-9&.+-]*){0,4})\s+"
    r"(?:(?:has|have|just)\s+)?(?:announced|launched|released|introduced|unveiled|published|open-sourced)\b"
)
_RULES = """
PRODUCTION EDITORIAL RULES:
- Tell ONE coherent, current story. One source may be the main evidence and another may add directly relevant context; never splice unrelated announcements into a broad trend.
- The title and opening sentence must name a concrete company, product, policy, model, benchmark, or release from the supplied source titles.
- Distinguish the article authority from the hosting publisher. Never say a feed or model-hosting platform launched or created something unless the supplied evidence explicitly says it did.
- Article author/byline metadata is provenance, not factual evidence about the reported project. Never claim an author led, owned, founded, announced, built, or was responsible for the project unless that role is explicitly supported in the supplied title or summary.
- Open with the specific change and its consequence in the first 12 words. Do not begin with a generic industry overview.
- Narration must be 130-140 words so the finished vertical video lands inside 55-62 seconds after natural pauses.
- Avoid generic filler such as “AI advancements,” “leading advancements,” “shaping the future,” and “more effective, ethical, and accessible.”
- Never describe selected publishers as collaborating, partnering, jointly developing, or confirming one another unless a supplied source explicitly states that relationship.
- Every scene must add a distinct fact or implication. Do not repeat the same conclusion in different wording.
- Use at least four concrete facts from the supplied title and summary. When the evidence contains measurements, preserve at least two of them exactly; do not replace context size, memory, CPU/GPU speed, or other measured capabilities with generic trend language.
- Never select an entry whose summary is empty or too weak to support four concrete facts.
- End with a source-backed limitation, comparison, or verification step; do not restate the model size, efficiency, accessibility, or workflow benefit.
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
    authority = source_authority(primary)
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
        title = f"{authority}: {subject}"
    if len(title) < 28:
        title = f"{title} — What Changed"
    title = title[:78].rstrip(" -—:;,.")

    narration = edit(package.narration)
    sentences = re.split(r"(?<=[.!?])\s+", narration.strip(), maxsplit=1)
    if sentences and not (_tokens(sentences[0]) & source_specific):
        concrete_hook = f"{authority} just detailed {subject}, and the change matters now."
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

    copy = " ".join(
        (
            package.title,
            package.narration,
            package.description,
            package.top_comment,
            *(scene.heading for scene in package.scenes),
            *(scene.body for scene in package.scenes),
        )
    ).casefold()
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


def _authority_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in {"ai", "the", "team", "labs", "lab"}
    }


def _validate_release_authority(
    package: VideoPackage,
    selected: list[SourceItem],
) -> None:
    """Reject a host/publisher claiming an announcement owned by another authority."""
    from .local_llm import LocalLLMError

    if not selected:
        return
    primary = selected[0]
    authority = source_authority(primary)
    if not authority or authority.casefold() == primary.publisher.casefold():
        return
    expected = _authority_tokens(authority)
    publisher = _authority_tokens(primary.publisher)
    opening = re.split(r"(?<=[.!?])\s+", package.narration.strip(), maxsplit=1)[0]
    for label, text in (("title", package.title), ("opening sentence", opening)):
        match = _RELEASE_ACTOR_RE.search(text)
        if not match:
            continue
        actor = match.group(1)
        actor_tokens = _authority_tokens(actor)
        if actor_tokens & publisher and not actor_tokens & expected:
            raise LocalLLMError(
                f"Production {label} attributes the release to hosting publisher "
                f"{primary.publisher!r}; supplied source authority is {authority!r}"
            )


_AUTHOR_ORG_MARKERS = {
    "ai", "research", "labs", "lab", "team", "staff", "newsroom", "editorial",
    "press", "group", "foundation", "university", "institute", "inc", "llc", "corp",
    "corporation", "company", "technologies", "technology",
}


def _personal_source_author(source: SourceItem) -> str:
    author = " ".join(str(getattr(source, "author", "") or "").split()).strip()
    if not author or author.casefold() == str(source.publisher).strip().casefold():
        return ""
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", author)
    if not 2 <= len(tokens) <= 4:
        return ""
    lowered = {token.casefold().strip(".-'") for token in tokens}
    if lowered & _AUTHOR_ORG_MARKERS:
        return ""
    if not all(token[0].isupper() for token in tokens if token):
        return ""
    return author


def _validate_author_metadata_grounding(
    package: VideoPackage,
    selected: list[SourceItem],
) -> None:
    """Prevent a byline from becoming an invented project role.

    A personal author may appear in viewer copy only when the selected article's factual title or
    summary also mentions that person. This intentionally treats author metadata as provenance,
    never as evidence of leadership, ownership, responsibility, or announcement authority.
    """
    from .local_llm import LocalLLMError

    viewer_copy = " ".join(
        (
            package.title,
            package.narration,
            package.description,
            package.top_comment,
            *(scene.heading for scene in package.scenes),
            *(scene.body for scene in package.scenes),
        )
    ).casefold()
    for source in selected:
        author = _personal_source_author(source)
        if not author or author.casefold() not in viewer_copy:
            continue
        factual_evidence = f"{source.title} {source.summary}".casefold()
        if author.casefold() not in factual_evidence:
            raise LocalLLMError(
                "Production viewer copy promotes article author/byline metadata into an unsupported "
                f"factual claim about {author!r}; the selected title/summary does not mention that person"
            )


def _evidence_numbers(value: str) -> set[str]:
    numbers = {
        match.replace(",", "").casefold()
        for match in re.findall(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)", value)
    }
    return {
        number
        for number in numbers
        if not (number.isdigit() and 2000 <= int(number) <= 2099)
    }


def _validate_evidence_specificity(
    package: VideoPackage,
    selected: list[SourceItem],
) -> None:
    """Require available measured evidence to survive into the publishable script."""
    from .local_llm import LocalLLMError

    if not selected:
        return
    weak = [source.url for source in selected if len(source.summary.split()) < 40]
    if weak:
        raise LocalLLMError(
            "Selected source evidence is too thin for a factual production script: "
            + ", ".join(weak)
        )
    evidence_numbers: set[str] = set()
    for source in selected:
        evidence_numbers.update(
            _evidence_numbers(source.summary) - _evidence_numbers(source.title)
        )
    if not evidence_numbers:
        return
    copy = " ".join(
        (
            package.title,
            package.narration,
            *(scene.body for scene in package.scenes),
        )
    )
    preserved = evidence_numbers & _evidence_numbers(copy)
    required = min(2, len(evidence_numbers))
    if len(preserved) < required:
        raise LocalLLMError(
            "Production copy replaced measured source evidence with generic language; "
            f"requires {required} supplied measurement(s), preserved {sorted(preserved)}"
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
    if not 130 <= word_count <= 140:
        raise LocalLLMError(
            f"Production narration must contain 130-140 words; received {word_count}"
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
    _validate_release_authority(package, selected)
    _validate_author_metadata_grounding(package, selected)
    _validate_evidence_specificity(package, selected)

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
