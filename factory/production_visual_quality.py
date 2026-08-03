from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image


_INSTALLED = False
_MAX_KEYFRAME_ATTEMPTS = 3
_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_BRAND_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bMicrosoft\s+Research\b", re.I), "a research team"),
    (re.compile(r"\bGoogle\s+AI\b", re.I), "an independent research team"),
    (re.compile(r"\bOpenAI\b", re.I), "an AI research team"),
    (re.compile(r"\bNVIDIA\b", re.I), "an edge-computing team"),
    (re.compile(r"\bEvoLib\b", re.I), "an adaptive knowledge system"),
    (re.compile(r"\bJetson\b", re.I), "a compact edge-computing module"),
    (re.compile(r"\bARC-AGI-3\b", re.I), "a reasoning benchmark"),
)
_EXTRA_NEGATIVE = (
    "text, words, letters, numbers, typography, logo, watermark, signage, printed page, "
    "collage, triptych, contact sheet, magazine layout, poster, infographic, panel grid, "
    "split screen, frame within frame, phone, smartphone, tablet, device mockup, monitor, "
    "interface, duplicated subject, malformed anatomy"
)


@dataclass(frozen=True)
class KeyframeQualityFinding:
    scene_index: int
    attempt: int
    detected_text: tuple[str, ...]
    layout_flags: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.detected_text and not self.layout_flags

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def _single_view_director_prompt(value: str) -> str:
    result = re.sub(
        r"Factual\s+visual:\s*[^;]+;\s*[^;]+;\s*",
        "Factual visual: ",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in _BRAND_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    result = re.sub(
        r"\b(?:phone|smartphone|tablet|monitor|screen|dashboard|interface|poster|magazine)\b",
        "human-scale workspace",
        result,
        flags=re.IGNORECASE,
    )
    return result


def _ocr_tokens(image: Image.Image) -> tuple[str, ...]:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Production keyframe OCR requires pytesseract") from exc
    data = pytesseract.image_to_data(
        image.convert("RGB"),
        output_type=pytesseract.Output.DICT,
        config="--psm 11",
    )
    strong: list[str] = []
    weak: list[str] = []
    for raw_text, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
        text = " ".join(str(raw_text).split()).strip()
        clean = _ALNUM_RE.sub("", text)
        if len(clean) < 2:
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= 45.0 and len(clean) >= 3:
            strong.append(text)
        elif confidence >= 18.0:
            weak.append(text)
    if strong:
        return tuple(dict.fromkeys(strong + weak))
    if len(weak) >= 3:
        return tuple(dict.fromkeys(weak))
    return ()


def _band_count(values: Any, *, threshold: float, lower: float, upper: float) -> int:
    import numpy as np

    indices = np.where(values >= threshold)[0]
    if len(indices) == 0:
        return 0
    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index > previous + 1:
            groups.append((start, previous))
            start = index
        previous = index
    groups.append((start, previous))
    length = len(values)
    return sum(
        lower <= ((start + end) / 2) / max(1, length - 1) <= upper
        for start, end in groups
    )


def _layout_flags(image: Image.Image) -> tuple[str, ...]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Production keyframe layout review requires OpenCV and NumPy") from exc

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = gray < 45
    horizontal_bands = _band_count(
        dark.mean(axis=1), threshold=0.70, lower=0.08, upper=0.66
    )
    vertical_bands = _band_count(
        dark.mean(axis=0), threshold=0.70, lower=0.08, upper=0.92
    )
    flags: list[str] = []
    if horizontal_bands >= 2 or vertical_bands >= 2:
        flags.append("multi_panel_or_collage_layout")

    height, width = gray.shape
    mask = cv2.inRange(gray, 0, 65)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        if box_area <= 0:
            continue
        area_ratio = box_area / (width * height)
        width_ratio = box_width / width
        height_ratio = box_height / height
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        rectangularity = cv2.contourArea(contour) / box_area
        if (
            0.25 <= area_ratio <= 0.72
            and 0.42 <= width_ratio <= 0.78
            and 0.58 <= height_ratio <= 0.92
            and 0.40 <= center_x <= 0.60
            and 0.38 <= center_y <= 0.62
            and rectangularity >= 0.80
        ):
            flags.append("large_centered_device_like_frame")
            break
    return tuple(flags)


def inspect_keyframe(path: Path, *, scene_index: int, attempt: int) -> KeyframeQualityFinding:
    with Image.open(path) as image:
        image.load()
        return KeyframeQualityFinding(
            scene_index=scene_index,
            attempt=attempt,
            detected_text=_ocr_tokens(image),
            layout_flags=_layout_flags(image),
        )


def _retry_seed(seed: int, scene_index: int, attempt: int) -> int:
    return (seed + attempt * 104729 + scene_index * 1009) % 2_147_483_647


def _write_quality_report(
    output_dir: Path,
    findings: list[KeyframeQualityFinding],
    *,
    accepted_attempt: int | None,
) -> Path:
    report = {
        "schema_version": 1,
        "ocr_backend": "tesseract_psm11",
        "layout_review": "separator_bands_and_centered_device_contour",
        "max_attempts": _MAX_KEYFRAME_ATTEMPTS,
        "accepted_attempt": accepted_attempt,
        "passed": accepted_attempt is not None,
        "findings": [finding.as_dict() for finding in findings],
    }
    path = output_dir / "keyframe-quality-report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _reviewed_generate(
    original: Callable[..., Any],
    generator: Any,
    output_dir: Path,
) -> Any:
    from .image_generator import ImageGenerationError

    original_plan = generator.plan
    current_plan = original_plan
    all_findings: list[KeyframeQualityFinding] = []
    try:
        for attempt in range(1, _MAX_KEYFRAME_ATTEMPTS + 1):
            generator.plan = current_plan
            assets = original(generator, output_dir)
            findings = [
                inspect_keyframe(
                    asset.path,
                    scene_index=asset.scene_index,
                    attempt=attempt,
                )
                for asset in assets
            ]
            all_findings.extend(findings)
            rejected = {finding.scene_index for finding in findings if not finding.passed}
            if not rejected:
                _write_quality_report(output_dir, all_findings, accepted_attempt=attempt)
                return assets
            if attempt >= _MAX_KEYFRAME_ATTEMPTS:
                _write_quality_report(output_dir, all_findings, accepted_attempt=None)
                details = "; ".join(
                    f"scene {finding.scene_index}: text={list(finding.detected_text)}, "
                    f"layout={list(finding.layout_flags)}"
                    for finding in findings
                    if not finding.passed
                )
                raise ImageGenerationError(
                    "Generated keyframes failed text/layout review after "
                    f"{_MAX_KEYFRAME_ATTEMPTS} attempts: {details}"
                )
            repaired_scenes = tuple(
                replace(
                    scene,
                    seed=_retry_seed(scene.seed, scene.scene_index, attempt),
                )
                if scene.scene_index in rejected
                else scene
                for scene in current_plan.scenes
            )
            current_plan = replace(current_plan, scenes=repaired_scenes)
    finally:
        generator.plan = original_plan


def install_production_visual_quality() -> None:
    """Install text-free single-view prompt compilation and keyframe retries."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import image_generator
    from .visual_prompt_compiler import CompiledVisualPrompt

    original_compiler = image_generator.compile_image_prompt

    def single_view_compiler(
        director_prompt: str,
        director_negative_prompt: str = "",
        **kwargs: Any,
    ) -> CompiledVisualPrompt:
        executable = original_compiler(
            _single_view_director_prompt(director_prompt),
            f"{_EXTRA_NEGATIVE}, {director_negative_prompt}",
            **kwargs,
        )
        prompt = executable.compiled_prompt.replace(
            "Text-free cinematic editorial image.",
            "Single continuous cinematic photograph, one camera view, text-free.",
            1,
        )
        prompt = prompt.replace(
            "No screens, signs, symbols, logos, or interfaces.",
            "Natural physical scene, coherent perspective.",
            1,
        )
        return replace(
            executable,
            director_prompt=director_prompt,
            compiled_prompt=prompt,
            word_count=len(prompt.split()),
            negative_prompt=_EXTRA_NEGATIVE,
            compiler_version="visual-compiler-v6-perceptual",
        )

    image_generator.compile_image_prompt = single_view_compiler
    original_sdxl = image_generator.SDXLLightningKeyframeGenerator.generate
    original_flux = image_generator.FluxKeyframeGenerator.generate

    def reviewed_sdxl(self: Any, output_dir: Path) -> Any:
        return _reviewed_generate(original_sdxl, self, output_dir)

    def reviewed_flux(self: Any, output_dir: Path) -> Any:
        return _reviewed_generate(original_flux, self, output_dir)

    image_generator.SDXLLightningKeyframeGenerator.generate = reviewed_sdxl
    image_generator.FluxKeyframeGenerator.generate = reviewed_flux
    _INSTALLED = True
