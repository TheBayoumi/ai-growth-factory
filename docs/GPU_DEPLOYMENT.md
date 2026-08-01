# Real GPU deployment

## Production target: Modal T4

The repository deploys one bounded T4 worker from `cloud/modal_app.py`.
The models run sequentially to stay within the 16 GB VRAM envelope:

1. Qwen3-4B through llama.cpp creates the factual package.
2. The llama process stops and releases GPU memory.
3. Qwen3-TTS generates narration segments.
4. TTS unloads.
5. Qwen2.5-Omni GPTQ Int4 reviews each segment with text-only output.
6. Rejected segments are regenerated once.
7. FFmpeg renders the final private canary.
8. Audio provenance, perceptual review and temporal-stability gates must pass before upload.

## Required Modal secret

Create `ai-growth-factory-secrets` in Modal with:

- `YOUTUBE_OAUTH_JSON`
- `PUBLISH_ENABLED=true`
- `YOUTUBE_PRIVACY_STATUS=private`
- optional `HF_TOKEN` when a model requires authenticated download

Never commit these values.

## GitHub deployment secrets

Create the protected GitHub environment `modal-production` and add:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Use environment reviewers when available. Deployment is manual by default. The workflow does not deploy on every commit.

## Deploy and canary

Run **Deploy Modal GPU Worker** from GitHub Actions. Enable `run_private_canary` only after the Modal secret exists.

The independent **Run Private GPU Canary** workflow invokes the deployed image without changing source.

## Promotion policy

Keep uploads private until three consecutive real-model canaries pass perceptual inspection. Test fixtures, eSpeak audio and muted visual checks are never production canaries.

## Repository creation fallback

The ChatGPT GitHub connector can push files and commits to an existing repository but does not expose GitHub's repository-creation endpoint. The included `scripts/create_and_push_github.ps1` and `.sh` perform the one-time creation with GitHub CLI, after which the connector and workflows can operate normally.
