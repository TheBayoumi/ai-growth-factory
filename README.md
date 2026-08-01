# AI Growth Factory — Free-First Cloud Edition

Version **1.3.1** is the production source release for an autonomous, private-first AI video pipeline.

## Architecture

```text
Modal cron (10:00 Africa/Cairo)
  → primary-source trend research
  → managed llama.cpp / Qwen3-4B GGUF
  → Qwen3-TTS segmented narration
  → deterministic audio QC
  → Qwen2.5-Omni GPTQ Int4 perceptual review
  → selective segment regeneration
  → stable FFmpeg portrait render
  → temporal/video QC
  → private-first YouTube upload
  → analytics feedback and subscriber-aware Thompson sampling
```

The models run serially so they are not expected to remain resident together on a T4.

## Safety and quality gates

Publication fails closed unless all gates pass:

1. Fresh primary-source evidence from at least two publishers.
2. Qwen3-TTS generator provenance.
3. Objective clipping, RMS, silence, pace and DC-offset checks.
4. Perceptual reviewer approval and local score thresholds.
5. Stable render, codec, duration, thumbnail and temporal-motion checks.
6. Correct source labels for every scene.
7. Private-first YouTube status during validation.

## CI policy

CI is triggered **only when a pull request is closed as merged into `main`**. It does not run on pull-request edits or ordinary branch pushes.

## Local verification

```bash
python -m pip install -e . coverage
python -m compileall -q factory api tests cloud
coverage run --branch -m unittest discover -s tests -v
coverage report --fail-under=65
python scripts/repository_preflight.py
```

## Modal deployment

Create the protected GitHub environment `modal-production`, add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, then run `Deploy Modal GPU Worker` manually. Runtime YouTube credentials belong in the Modal secret `ai-growth-factory-secrets`, never in GitHub source.

The scheduled worker is bounded to one T4 container, a thirty-minute timeout, and one infrastructure retry. The first three uploads remain private for owner review.
