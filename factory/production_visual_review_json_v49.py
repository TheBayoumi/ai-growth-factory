from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator


_INSTALLED = False
_MAX_SYNTAX_REPAIRS = 2
_MAX_REVIEW_PASSES = 2
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$", re.IGNORECASE)
_NEXT_KEY_RE = re.compile(r'^"(?:[^"\\]|\\.)+"\s*:')


def _strip_fence(text: str) -> str:
    clean = str(text or "").strip()
    if clean.startswith("```"):
        clean = _FENCE_OPEN_RE.sub("", clean)
        clean = _FENCE_CLOSE_RE.sub("", clean)
    return clean.strip()


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield brace-balanced JSON-object candidates without interpreting their values."""
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
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
                yield text[start : index + 1]
                start = None


def _remove_trailing_commas(candidate: str) -> str:
    # JSON-only syntax normalization. Values and key text are untouched.
    return re.sub(r",\s*([}\]])", r"\1", candidate)


def _insert_missing_field_comma(candidate: str, error: json.JSONDecodeError) -> str | None:
    """Repair only the narrow `value <whitespace> "next_key":` serialization defect."""
    if error.msg != "Expecting ',' delimiter":
        return None
    position = max(0, min(len(candidate), int(error.pos)))
    prefix = candidate[:position]
    suffix = candidate[position:]
    stripped_prefix = prefix.rstrip()
    stripped_suffix = suffix.lstrip()
    if not stripped_prefix or not stripped_suffix or not _NEXT_KEY_RE.match(stripped_suffix):
        return None

    previous = stripped_prefix[-1]
    previous_is_complete_value = (
        previous in {'"', '}', ']'}
        or previous.isdigit()
        or stripped_prefix.endswith(("true", "false", "null"))
    )
    if not previous_is_complete_value:
        return None

    # Preserve whitespace exactly and insert only the required field separator.
    insert_at = len(stripped_prefix)
    return candidate[:insert_at] + "," + candidate[insert_at:]


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
    current = candidate.strip()
    for repair_index in range(_MAX_SYNTAX_REPAIRS + 1):
        try:
            value = json.loads(current)
        except json.JSONDecodeError as exc:
            repaired = _remove_trailing_commas(current)
            if repaired != current:
                current = repaired
                continue
            if repair_index >= _MAX_SYNTAX_REPAIRS:
                return None
            repaired = _insert_missing_field_comma(current, exc)
            if repaired is None or repaired == current:
                return None
            current = repaired
            continue
        if isinstance(value, dict):
            return value
        return None
    return None


def _malformed_error(clean: str, detail: str = "") -> Exception:
    from .production_visual_quality import VisualQualityError

    digest = hashlib.sha256(clean.encode("utf-8", errors="replace")).hexdigest()
    excerpt = " ".join(clean.split())[:240]
    suffix = f"; {detail}" if detail else ""
    return VisualQualityError(
        "Visual reviewer returned malformed JSON after bounded syntax recovery; "
        f"response_sha256={digest}; excerpt={excerpt!r}{suffix}"
    )


def extract_visual_review_json_v49(text: str) -> dict[str, Any]:
    """Parse reviewer JSON with bounded syntax-only recovery and fail closed otherwise."""
    clean = _strip_fence(text)

    # A syntactically valid top-level payload has authoritative shape. Never tunnel into an
    # array and approve an object nested inside it; the reviewer contract requires one object.
    try:
        top_level = json.loads(clean)
    except json.JSONDecodeError:
        top_level = None
    else:
        if isinstance(top_level, dict):
            return top_level
        raise _malformed_error(clean, "top-level reviewer JSON must be an object")

    candidates: list[str] = [clean]
    candidates.extend(candidate for candidate in _balanced_objects(clean) if candidate != clean)
    for candidate in candidates:
        value = _parse_candidate(candidate)
        if value is not None:
            return value

    raise _malformed_error(clean)


def _is_malformed_json_error(exc: Exception) -> bool:
    return str(exc).startswith("Visual reviewer returned malformed JSON")


def install_production_visual_review_json_v49() -> None:
    """Harden model-output parsing without changing any visual-quality threshold."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_visual_quality
    from .production_visual_quality import VisualQualityError
    from .production_visual_semantic_review_v28 import SemanticVisualReviewerV28

    production_visual_quality._extract_json = extract_visual_review_json_v49
    original_review = SemanticVisualReviewerV28.review

    def review_with_bounded_json_retry(self, *args, **kwargs):
        last_error: VisualQualityError | None = None
        for _pass in range(_MAX_REVIEW_PASSES):
            try:
                return original_review(self, *args, **kwargs)
            except VisualQualityError as exc:
                if not _is_malformed_json_error(exc):
                    raise
                last_error = exc
        assert last_error is not None
        raise VisualQualityError(
            "Visual reviewer could not serialize valid JSON after "
            f"{_MAX_REVIEW_PASSES} bounded passes: {last_error}"
        ) from last_error

    SemanticVisualReviewerV28.review = review_with_bounded_json_retry
    production_visual_quality._OmniVisualReviewer = SemanticVisualReviewerV28
    _INSTALLED = True
