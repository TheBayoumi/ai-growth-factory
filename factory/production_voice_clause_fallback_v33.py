from __future__ import annotations

import json
import re
import shutil
import wave
from pathlib import Path
from typing import Sequence

from .video_profile import VideoProfile


_INSTALLED = False
_UNREACHABLE_ERROR = "no candidate reachable within the v28 1.15x tempo ceiling"
_CONNECTORS = {
    "and",
    "but",
    "because",
    "while",
    "which",
    "that",
    "so",
    "as",
    "by",
}


def split_sentence_for_clause_fallback_v33(text: str) -> tuple[str, ...]:
    """Split one unreachable sentence at the strongest natural midpoint boundary.

    The normalized transcript is preserved exactly when the returned clauses are joined by one
    space. Punctuation boundaries outrank conjunctions; a balanced word boundary is the final
    fallback. Very short text remains unsplit so it still fails closed rather than sounding choppy.
    """
    clean = " ".join(text.split()).strip()
    words = clean.split()
    if len(words) < 10:
        return (clean,)

    minimum_side = max(4, min(7, len(words) // 3))
    midpoint = len(words) / 2.0
    candidates: list[tuple[int, float, int]] = []
    for index in range(minimum_side, len(words) - minimum_side + 1):
        previous = words[index - 1]
        current = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", words[index]).casefold()
        punctuation_priority = 0 if previous.endswith((",", ";", ":")) else 1
        connector_priority = 0 if current in _CONNECTORS else 1
        if punctuation_priority == 0 or connector_priority == 0:
            candidates.append(
                (punctuation_priority + connector_priority, abs(index - midpoint), index)
            )

    if candidates:
        split_at = min(candidates)[2]
    else:
        split_at = round(midpoint)
        split_at = max(minimum_side, min(len(words) - minimum_side, split_at))

    clauses = (" ".join(words[:split_at]), " ".join(words[split_at:]))
    if any(len(clause.split()) < minimum_side for clause in clauses):
        return (clean,)
    if " ".join(clauses) != clean:
        raise ValueError("Clause fallback changed the supplied transcript")
    return clauses


def join_pcm_wavs_v33(
    paths: Sequence[Path],
    output_path: Path,
    *,
    pause_ms: int = 120,
) -> Path:
    """Join compatible PCM WAV clauses with one deterministic natural pause."""
    if len(paths) < 2:
        raise ValueError("Clause fallback requires at least two WAV assets")
    if not 40 <= pause_ms <= 220:
        raise ValueError("Clause fallback pause must be between 40 and 220 ms")

    parameters: tuple[int, int, int] | None = None
    payloads: list[bytes] = []
    for path in paths:
        with wave.open(str(path), "rb") as handle:
            current = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if handle.getcomptype() != "NONE":
                raise ValueError(f"Clause WAV must be uncompressed PCM: {path}")
            if parameters is None:
                parameters = current
            elif current != parameters:
                raise ValueError("Clause WAV assets have incompatible audio parameters")
            payloads.append(handle.readframes(handle.getnframes()))

    assert parameters is not None
    channels, sample_width, sample_rate = parameters
    pause_frames = round(sample_rate * pause_ms / 1000.0)
    silence = b"\x00" * pause_frames * channels * sample_width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        for index, payload in enumerate(payloads):
            if index:
                handle.writeframes(silence)
            handle.writeframes(payload)
    return output_path


def _diagnostic_root(output_path: Path) -> Path:
    return output_path.parent.parent if output_path.parent.name == "segments" else output_path.parent


def _write_failure_diagnostic(
    output_path: Path,
    *,
    text: str,
    clauses: Sequence[str],
    trigger_error: str,
    fallback_error: str,
    events: Sequence[dict[str, object]],
) -> Path:
    destination = _diagnostic_root(output_path) / "voice-calibration-failure.json"
    payload = {
        "status": "voice_clause_fallback_failed_closed",
        "requested_output": str(output_path),
        "text": text,
        "word_count": len(text.split()),
        "clauses": list(clauses),
        "trigger_error": trigger_error,
        "fallback_error": fallback_error,
        "calibration_events": [dict(event) for event in events],
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def build_clause_fallback_tts_class_v33(
    base_tts: type,
    *,
    profile: VideoProfile,
) -> type:
    """Wrap the convergent TTS class with a bounded clause-level recovery path."""
    from . import production_voice_calibration_v28 as calibration
    from . import qwen_tts

    class ClauseFallbackQwen3TTS(base_tts):
        def generate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
        ) -> Path:
            event_start = len(calibration._CALIBRATION_EVENTS)
            try:
                return super().generate(
                    text=text,
                    instruction=instruction,
                    output_path=output_path,
                    seed=seed,
                )
            except qwen_tts.QwenTTSError as exc:
                trigger = str(exc)
                if _UNREACHABLE_ERROR not in trigger.casefold():
                    raise

            clauses = split_sentence_for_clause_fallback_v33(text)
            if len(clauses) < 2:
                events = calibration._CALIBRATION_EVENTS[event_start:]
                _write_failure_diagnostic(
                    output_path,
                    text=text,
                    clauses=clauses,
                    trigger_error=trigger,
                    fallback_error="Sentence is too short for safe clause decomposition",
                    events=events,
                )
                raise qwen_tts.QwenTTSError(trigger)

            temporary_paths: list[Path] = []
            compacted_paths: list[Path] = []
            clause_details: list[dict[str, object]] = []
            joined = output_path.with_name(f"{output_path.stem}-clause-joined{output_path.suffix}")
            try:
                for index, clause in enumerate(clauses):
                    raw_clause = output_path.with_name(
                        f"{output_path.stem}-clause-{index + 1}-raw{output_path.suffix}"
                    )
                    compacted_clause = output_path.with_name(
                        f"{output_path.stem}-clause-{index + 1}-compacted{output_path.suffix}"
                    )
                    clause_instruction = (
                        instruction.rstrip(" .")
                        + ". Synthesize only this clause as a continuous part of one longer "
                        "sentence. Begin promptly, leave no trailing pause, target a natural "
                        "146 to 150 words per minute before post-processing, preserve every "
                        "supplied word, and do not add an introduction or conclusion."
                    )
                    clause_seed = (int(seed) + (index + 1) * 0x27D4EB2D) & 0xFFFFFFFF
                    super().generate(
                        text=clause,
                        instruction=clause_instruction,
                        output_path=raw_clause,
                        seed=clause_seed,
                    )
                    compaction = calibration.compact_excess_silence_v28(
                        raw_clause,
                        compacted_clause,
                        maximum_internal_silence_ms=180,
                        maximum_edge_silence_ms=20,
                    )
                    observed = (
                        len(clause.split())
                        / max(float(compaction["after_seconds"]), 0.001)
                        * 60.0
                    )
                    if not calibration.segment_candidate_reachable_v28(
                        observed,
                        profile=profile,
                    ):
                        raise qwen_tts.QwenTTSError(
                            f"Clause {index + 1} remained unreachable at {observed:.2f} WPM"
                        )
                    temporary_paths.extend((raw_clause, compacted_clause))
                    compacted_paths.append(compacted_clause)
                    clause_details.append(
                        {
                            "clause_index": index,
                            "text": clause,
                            "word_count": len(clause.split()),
                            "seed": clause_seed,
                            "observed_compacted_wpm": round(observed, 3),
                            "removed_silence_seconds": compaction["removed_seconds"],
                        }
                    )

                join_pcm_wavs_v33(compacted_paths, joined, pause_ms=120)
                final_compaction = calibration.compact_excess_silence_v28(
                    joined,
                    output_path,
                    maximum_internal_silence_ms=180,
                    maximum_edge_silence_ms=40,
                )
                combined_wpm = (
                    len(text.split())
                    / max(float(final_compaction["after_seconds"]), 0.001)
                    * 60.0
                )
                if not calibration.segment_candidate_reachable_v28(
                    combined_wpm,
                    profile=profile,
                ):
                    raise qwen_tts.QwenTTSError(
                        f"Combined clause fallback remained unreachable at {combined_wpm:.2f} WPM"
                    )
                calibration._CALIBRATION_EVENTS.append(
                    {
                        "type": "bounded_clause_fallback_v33",
                        "requested_output": str(output_path),
                        "trigger_error": trigger,
                        "clause_count": len(clauses),
                        "join_pause_ms": 120,
                        "combined_observed_wpm": round(combined_wpm, 3),
                        "removed_joined_silence_seconds": final_compaction["removed_seconds"],
                        "clauses": clause_details,
                        "reachable": True,
                        "selected": True,
                    }
                )
                return output_path
            except Exception as fallback_exc:
                events = calibration._CALIBRATION_EVENTS[event_start:]
                _write_failure_diagnostic(
                    output_path,
                    text=text,
                    clauses=clauses,
                    trigger_error=trigger,
                    fallback_error=str(fallback_exc),
                    events=events,
                )
                output_path.unlink(missing_ok=True)
                raise qwen_tts.QwenTTSError(
                    f"{trigger}; bounded clause fallback failed: {fallback_exc}"
                ) from fallback_exc
            finally:
                joined.unlink(missing_ok=True)
                for path in temporary_paths:
                    path.unlink(missing_ok=True)

    ClauseFallbackQwen3TTS.__name__ = "ClauseFallbackQwen3TTS"
    return ClauseFallbackQwen3TTS


def install_production_voice_clause_fallback_v33() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import canary, voice_pipeline

    profile = VideoProfile.from_env()
    voice_pipeline.Qwen3TTS = build_clause_fallback_tts_class_v33(
        voice_pipeline.Qwen3TTS,
        profile=profile,
    )

    original_copy_voice_diagnostics = canary._copy_voice_diagnostics

    def copy_voice_diagnostics_with_calibration_failure(
        workdir: Path,
        destination: Path,
    ) -> None:
        failure = workdir / "voice-calibration-failure.json"
        if failure.is_file():
            canary._copy(failure, destination, "voice-calibration-failure.json")
        original_copy_voice_diagnostics(workdir, destination)

    canary._copy_voice_diagnostics = copy_voice_diagnostics_with_calibration_failure
    _INSTALLED = True
