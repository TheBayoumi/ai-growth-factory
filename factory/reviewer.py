from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import requests

from .config import Settings
from .models import AudioMetrics, AudioReview, NarrationSegment, VoiceContract


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("reviewer returned no JSON")
        return json.loads(text[start:end + 1])


def review_passes(review: AudioReview, settings: Settings) -> bool:
    return review.decision == "approve" and review.overall_score >= settings.reviewer_overall_threshold and review.scores.script_fidelity >= settings.reviewer_fidelity_threshold and review.scores.naturalness >= settings.reviewer_naturalness_threshold and review.scores.pronunciation >= settings.reviewer_pronunciation_threshold and review.scores.audio_artifacts >= 0.90 and not review.failed_segments


class OpenAIReviewer:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the optional OpenAI reviewer")
        self.settings = settings

    def review(self, audio_path: Path, narration: str, contract: VoiceContract, segments: list[NarrationSegment], metrics: AudioMetrics, attempt: int) -> AudioReview:
        prompt = {"task": "Assess exact script fidelity and narration quality. Return JSON only.", "attempt": attempt, "transcript": narration, "contract": contract.as_dict(), "metrics": metrics.as_dict(), "segments": [{"segment_id": item.segment_id, "text": item.text} for item in segments]}
        response = requests.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"}, json={"model": self.settings.openai_reviewer_model, "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}, {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio_path.read_bytes()).decode(), "format": "wav"}}]}], "store": False}, timeout=180)
        response.raise_for_status()
        data = response.json()
        text = data.get("output_text") or ""
        return AudioReview.from_dict(extract_json(text), self.settings.openai_reviewer_model, text)

    def unload(self) -> None:
        return None
