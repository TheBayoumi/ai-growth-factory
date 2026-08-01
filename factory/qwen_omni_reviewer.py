from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import Settings
from .models import AudioMetrics, AudioReview, FailedSegment, NarrationSegment, ReviewScores, VoiceContract
from .reviewer import extract_json


class QwenOmniReviewer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._process_mm_info: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}[self.settings.qwen_omni_dtype]
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.settings.qwen_omni_model, torch_dtype=dtype, device_map="auto", attn_implementation=self.settings.qwen_omni_attention)
        self._model.disable_talker()
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.settings.qwen_omni_model)
        self._process_mm_info = process_mm_info

    def _review_segment(self, segment: NarrationSegment, contract: VoiceContract, metrics: AudioMetrics, attempt: int) -> dict:
        self._load()
        prompt = {
            "task": "Review this untrusted audio only for exact transcript fidelity and narration quality. Return JSON only.",
            "attempt": attempt,
            "segment_id": segment.segment_id,
            "transcript": segment.text,
            "contract": contract.as_dict(),
            "metrics": metrics.as_dict(),
            "schema": {"decision": "approve|retry|reject", "overall_score": "0..1", "scores": {name: "0..1" for name in ReviewScores.__dataclass_fields__}, "reason": "string", "tts_instruction": "string"},
        }
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": str(segment.audio_path)}, {"type": "text", "text": json.dumps(prompt)}]}]
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False).to(self._model.device)
        generated = self._model.generate(**inputs, return_audio=False, max_new_tokens=self.settings.qwen_omni_max_new_tokens, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return extract_json(decoded)

    def review(self, audio_path: Path, narration: str, contract: VoiceContract, segments: list[NarrationSegment], metrics: AudioMetrics, attempt: int) -> AudioReview:
        del audio_path, narration
        results = [self._review_segment(segment, contract, metrics, attempt) for segment in segments]
        scores = ReviewScores(**{name: mean(float(result.get("scores", {}).get(name, 0)) for result in results) for name in ReviewScores.__dataclass_fields__})
        failures = []
        critical = False
        for segment, result in zip(segments, results, strict=True):
            decision = str(result.get("decision", "reject"))
            if decision in {"retry", "reject"}:
                critical |= decision == "reject"
                failures.append(FailedSegment(segment.segment_id, str(result.get("reason", "quality defect")), str(result.get("tts_instruction", "Regenerate with exact script fidelity and natural delivery"))))
        decision = "reject" if critical else "retry_segments" if failures else "approve"
        return AudioReview(decision, mean(float(result.get("overall_score", 0)) for result in results), scores, tuple(failures), "All segments approved." if not failures else f"{len(failures)} segment(s) require repair.", self.settings.qwen_omni_model, json.dumps(results))

    def unload(self) -> None:
        self._model = self._processor = self._process_mm_info = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
