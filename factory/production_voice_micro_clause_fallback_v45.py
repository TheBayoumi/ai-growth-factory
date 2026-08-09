from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .production_voice_clause_fallback_v33 import join_pcm_wavs_v33
from .production_voice_technical_identifier_v42 import (
    speech_equivalent_word_count_v42,
)
from .video_profile import VideoProfile


_INSTALLED = False
_UNREACHABLE_ERROR = "no candidate reachable within the v28 1.15x tempo ceiling"
_CONNECTORS = {"and", "but", "because", "while", "which", "that", "so", "as", "by"}


class MicroClauseFallbackError(RuntimeError):
    """The bounded transcript-exact micro-clause recovery could not reach publication pace."""


@dataclass(frozen=True)
class MicroClauseCandidate:
    text: str
    path: Path
    written_words: int
    speech_equivalent_words: float
    duration_seconds: float
    observed_wpm: float
    seed: int
    attempt: int
    removed_silence_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "written_words": self.written_words,
            "speech_equivalent_words": round(self.speech_equivalent_words, 3),
            "duration_seconds": round(self.duration_seconds, 6),
            "observed_wpm": round(self.observed_wpm, 3),
            "seed": self.seed,
            "attempt": self.attempt,
            "removed_silence_seconds": round(self.removed_silence_seconds, 6),
            "audio_path": str(self.path),
        }


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def split_micro_clause_v45(text: str) -> tuple[str, ...]:
    """Split an unreachable phrase into two exact, balanced four-word-or-longer pieces."""
    clean = " ".join(text.split()).strip()
    words = clean.split()
    if len(words) < 8:
        return (clean,)
    minimum_side = 4
    midpoint = len(words) / 2.0
    candidates: list[tuple[int, float, int]] = []
    for index in range(minimum_side, len(words) - minimum_side + 1):
        previous = words[index - 1]
        current = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", words[index]).casefold()
        punctuation_priority = 0 if previous.endswith((",", ";", ":")) else 1
        connector_priority = 0 if current in _CONNECTORS else 1
        candidates.append(
            (punctuation_priority + connector_priority, abs(index - midpoint), index)
        )
    split_at = min(candidates)[2]
    result = (" ".join(words[:split_at]), " ".join(words[split_at:]))
    if any(len(piece.split()) < minimum_side for piece in result):
        return (clean,)
    if " ".join(result) != clean:
        raise ValueError("Micro-clause splitting changed the supplied transcript")
    return result


def choose_piece_to_split_v45(
    pieces: Sequence[str],
    observed_wpm: Sequence[float],
    *,
    maximum_pieces: int = 4,
) -> int | None:
    """Choose the slowest safely splittable piece without exceeding the hard piece budget."""
    if len(pieces) != len(observed_wpm):
        raise ValueError("pieces and observed_wpm must have equal lengths")
    if len(pieces) >= maximum_pieces:
        return None
    candidates = [
        (float(observed_wpm[index]), -len(piece.split()), index)
        for index, piece in enumerate(pieces)
        if len(piece.split()) >= 8 and len(split_micro_clause_v45(piece)) == 2
    ]
    return min(candidates)[2] if candidates else None


def combined_micro_clause_wpm_v45(text: str, duration_seconds: float) -> float:
    return speech_equivalent_word_count_v42(text) / max(duration_seconds, 0.001) * 60.0


def _candidate_score(candidate: MicroClauseCandidate, profile: VideoProfile) -> tuple[float, float]:
    required = profile.target_wpm / max(candidate.observed_wpm, 0.001)
    bounded = min(max(required, 0.85), profile.maximum_tempo_factor)
    projected = candidate.observed_wpm * bounded
    reachable = candidate.observed_wpm * profile.maximum_tempo_factor >= profile.target_wpm - 2
    return (0.0 if reachable else 100.0) + abs(projected - profile.target_wpm), -candidate.observed_wpm


def _diagnostic_root(output_path: Path) -> Path:
    return output_path.parent.parent if output_path.parent.name == "segments" else output_path.parent


def _write_diagnostic(
    output_path: Path,
    *,
    text: str,
    trigger_error: str,
    pieces: Sequence[str],
    rounds: Sequence[dict[str, object]],
    error: str,
) -> Path:
    destination = _diagnostic_root(output_path) / "voice-micro-clause-failure.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "status": "voice_micro_clause_fallback_failed_closed",
                "requested_output": str(output_path),
                "text": text,
                "written_word_count": len(text.split()),
                "speech_equivalent_word_count": round(
                    speech_equivalent_word_count_v42(text), 3
                ),
                "trigger_error": trigger_error,
                "pieces": list(pieces),
                "rounds": list(rounds),
                "error": error,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destination


def build_micro_clause_fallback_tts_class_v45(
    base_tts: type,
    *,
    profile: VideoProfile,
) -> type:
    """Wrap the full v43 voice stack with one bounded raw micro-clause recovery."""
    from . import production_voice_calibration_v28 as calibration
    from . import qwen_tts

    class MicroClauseFallbackQwen3TTS(base_tts):
        def _generate_micro_candidate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
            piece_index: int,
            attempt: int,
        ) -> MicroClauseCandidate:
            raw = output_path.with_name(
                f"{output_path.stem}-micro-{piece_index + 1}-attempt-{attempt}-raw.wav"
            )
            compacted = output_path.with_name(
                f"{output_path.stem}-micro-{piece_index + 1}-attempt-{attempt}-compact.wav"
            )
            candidate_seed = (
                int(seed)
                + (piece_index + 1) * 0x165667B1
                + attempt * 0x27D4EB2D
            ) & 0xFFFFFFFF
            micro_instruction = (
                instruction.rstrip(" .")
                + ". Synthesize exactly this short phrase as one brisk, natural continuation "
                "inside a longer sentence. Start speaking immediately, use no introductory "
                "sound, no dramatic pause, and no trailing silence. Target approximately 180 "
                "words per minute before post-processing while preserving every supplied word, "
                "technical identifier, and punctuation. Do not repeat, omit, or paraphrase."
            )
            try:
                # Deliberately bypass the convergence wrappers for this bounded last-resort
                # candidate. Every raw take is measured and the joined result must still be
                # reachable under the unchanged 1.15x publication ceiling.
                qwen_tts.Qwen3TTS.generate(
                    self,
                    text=text,
                    instruction=micro_instruction,
                    output_path=raw,
                    seed=candidate_seed,
                )
                compaction = calibration.compact_excess_silence_v28(
                    raw,
                    compacted,
                    maximum_internal_silence_ms=120,
                    maximum_edge_silence_ms=10,
                )
                duration = float(compaction["after_seconds"])
                equivalent = speech_equivalent_word_count_v42(text)
                return MicroClauseCandidate(
                    text=text,
                    path=compacted,
                    written_words=len(text.split()),
                    speech_equivalent_words=equivalent,
                    duration_seconds=duration,
                    observed_wpm=equivalent / max(duration, 0.001) * 60.0,
                    seed=candidate_seed,
                    attempt=attempt,
                    removed_silence_seconds=float(compaction["removed_seconds"]),
                )
            finally:
                raw.unlink(missing_ok=True)

        def _best_micro_candidate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
            piece_index: int,
            attempts: int,
        ) -> tuple[MicroClauseCandidate, list[dict[str, object]]]:
            candidates: list[MicroClauseCandidate] = []
            errors: list[str] = []
            for attempt in range(1, attempts + 1):
                try:
                    candidates.append(
                        self._generate_micro_candidate(
                            text=text,
                            instruction=instruction,
                            output_path=output_path,
                            seed=seed,
                            piece_index=piece_index,
                            attempt=attempt,
                        )
                    )
                except Exception as exc:
                    errors.append(str(exc))
            if not candidates:
                raise MicroClauseFallbackError(
                    f"Micro-clause {piece_index + 1} produced no candidate: "
                    + "; ".join(errors[-2:])
                )
            candidates.sort(key=lambda item: _candidate_score(item, profile))
            selected = candidates[0]
            evidence = [
                {
                    **candidate.as_dict(),
                    "selected": candidate.path == selected.path,
                }
                for candidate in candidates
            ]
            for candidate in candidates[1:]:
                candidate.path.unlink(missing_ok=True)
            return selected, evidence

        def generate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
        ) -> Path:
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

            clean = " ".join(text.split()).strip()
            pieces = list(split_micro_clause_v45(clean))
            if len(pieces) < 2:
                raise qwen_tts.QwenTTSError(trigger)
            maximum_pieces = _env_int("V45_MAX_MICRO_CLAUSES", 4, 2, 4)
            attempts = _env_int("V45_MICRO_CLAUSE_ATTEMPTS", 3, 1, 5)
            rounds: list[dict[str, object]] = []
            selected_paths: list[Path] = []
            final_candidates: list[Path] = []
            joined_candidates: list[Path] = []
            try:
                round_index = 0
                while True:
                    round_index += 1
                    selected: list[MicroClauseCandidate] = []
                    candidate_evidence: list[dict[str, object]] = []
                    for piece_index, piece in enumerate(pieces):
                        candidate, evidence = self._best_micro_candidate(
                            text=piece,
                            instruction=instruction,
                            output_path=output_path,
                            seed=(int(seed) + round_index * 0x9E3779B1) & 0xFFFFFFFF,
                            piece_index=piece_index,
                            attempts=attempts,
                        )
                        selected.append(candidate)
                        selected_paths.append(candidate.path)
                        candidate_evidence.extend(evidence)

                    joined = output_path.with_name(
                        f"{output_path.stem}-micro-round-{round_index}-joined.wav"
                    )
                    final = output_path.with_name(
                        f"{output_path.stem}-micro-round-{round_index}-final.wav"
                    )
                    joined_candidates.append(joined)
                    final_candidates.append(final)
                    join_pcm_wavs_v33(
                        [candidate.path for candidate in selected],
                        joined,
                        pause_ms=50,
                    )
                    final_compaction = calibration.compact_excess_silence_v28(
                        joined,
                        final,
                        maximum_internal_silence_ms=120,
                        maximum_edge_silence_ms=20,
                    )
                    duration = float(final_compaction["after_seconds"])
                    combined_wpm = combined_micro_clause_wpm_v45(clean, duration)
                    reachable = calibration.segment_candidate_reachable_v28(
                        combined_wpm,
                        profile=profile,
                    )
                    rounds.append(
                        {
                            "round": round_index,
                            "pieces": list(pieces),
                            "piece_candidates": candidate_evidence,
                            "selected_piece_wpm": [
                                round(candidate.observed_wpm, 3)
                                for candidate in selected
                            ],
                            "join_pause_ms": 50,
                            "combined_duration_seconds": round(duration, 6),
                            "combined_observed_wpm": round(combined_wpm, 3),
                            "maximum_tempo_factor": profile.maximum_tempo_factor,
                            "projected_wpm_at_ceiling": round(
                                combined_wpm * profile.maximum_tempo_factor, 3
                            ),
                            "removed_joined_silence_seconds": final_compaction[
                                "removed_seconds"
                            ],
                            "reachable": reachable,
                        }
                    )
                    if reachable:
                        shutil.copy2(final, output_path)
                        calibration._CALIBRATION_EVENTS.append(
                            {
                                "type": "bounded_micro_clause_fallback_v45",
                                "requested_output": str(output_path),
                                "trigger_error": trigger,
                                "piece_count": len(pieces),
                                "pieces": list(pieces),
                                "rounds": rounds,
                                "combined_observed_wpm": round(combined_wpm, 3),
                                "maximum_tempo_factor": profile.maximum_tempo_factor,
                                "reachable": True,
                                "selected": True,
                            }
                        )
                        failure = _diagnostic_root(output_path) / "voice-calibration-failure.json"
                        failure.unlink(missing_ok=True)
                        micro_failure = (
                            _diagnostic_root(output_path)
                            / "voice-micro-clause-failure.json"
                        )
                        micro_failure.unlink(missing_ok=True)
                        return output_path

                    split_index = choose_piece_to_split_v45(
                        pieces,
                        [candidate.observed_wpm for candidate in selected],
                        maximum_pieces=maximum_pieces,
                    )
                    if split_index is None:
                        raise MicroClauseFallbackError(
                            f"Combined micro-clauses remained unreachable at "
                            f"{combined_wpm:.2f} WPM"
                        )
                    replacement = split_micro_clause_v45(pieces[split_index])
                    if len(replacement) != 2:
                        raise MicroClauseFallbackError(
                            f"Piece {split_index + 1} cannot be split safely"
                        )
                    pieces[split_index : split_index + 1] = list(replacement)
            except Exception as fallback_exc:
                _write_diagnostic(
                    output_path,
                    text=clean,
                    trigger_error=trigger,
                    pieces=pieces,
                    rounds=rounds,
                    error=str(fallback_exc),
                )
                output_path.unlink(missing_ok=True)
                raise qwen_tts.QwenTTSError(
                    f"{trigger}; bounded micro-clause fallback failed: {fallback_exc}"
                ) from fallback_exc
            finally:
                for path in [*selected_paths, *joined_candidates, *final_candidates]:
                    path.unlink(missing_ok=True)

    MicroClauseFallbackQwen3TTS.__name__ = "MicroClauseFallbackQwen3TTS"
    return MicroClauseFallbackQwen3TTS


def install_production_voice_micro_clause_fallback_v45() -> None:
    """Install after v42 so every raw piece uses speech-equivalent technical word counts."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import canary, voice_pipeline

    profile = VideoProfile.from_env()
    voice_pipeline.Qwen3TTS = build_micro_clause_fallback_tts_class_v45(
        voice_pipeline.Qwen3TTS,
        profile=profile,
    )

    original_copy_voice_diagnostics = canary._copy_voice_diagnostics

    def copy_voice_diagnostics_with_micro_failure(
        workdir: Path,
        destination: Path,
    ) -> None:
        failure = workdir / "voice-micro-clause-failure.json"
        if failure.is_file():
            canary._copy(failure, destination, "voice-micro-clause-failure.json")
        original_copy_voice_diagnostics(workdir, destination)

    canary._copy_voice_diagnostics = copy_voice_diagnostics_with_micro_failure
    _INSTALLED = True
