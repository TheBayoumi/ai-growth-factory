# AI Growth Factory — Free-First Cloud Edition

Version **1.3.1** is a corrective quality release. It invalidates the 1.3.0 media outputs, removes quantized zoom motion, and requires provable Qwen3-TTS plus perceptual-review approval before any video can be treated as a production canary.

## Selected proof-stage stack

| Layer | Selected service | Role |
|---|---|---|
| Autonomous production | Modal Starter | Daily T4 GPU job, secrets, cron and persistent model/state volumes |
| Interactive development | Lightning AI Free | Manual model testing, debugging and recovery workspace |
| Lightweight control plane | Vercel Hobby | Status and YouTube OAuth bootstrap only |
| Temporary scale experiment | Azure trial credit | Optional one-time benchmark after the private canaries work |
| Benchmark overflow | Kaggle notebooks | Manual experiments only; never the production scheduler |

Modal is the production choice because its free monthly compute credit, scheduled functions, secrets, persistent volumes and concurrency controls are compatible with an unattended worker. Lightning AI is useful for development, but free Studios restart and are not treated as durable production infrastructure. Azure's new-account credit expires after 30 days and its permanent free VMs are CPU-only, so Azure is not a sustainable GPU foundation.

## Autonomous architecture

```text
Modal daily cron — 10:00 Africa/Cairo
        ↓
T4 worker, maximum one container
        ↓
Primary-source RSS/Atom research and freshness gates
        ↓
Managed llama.cpp — Qwen3-4B Q4_K_M
        ↓
Unload script model and release VRAM
        ↓
Qwen3-TTS 0.6B — segmented narration
        ↓
Deterministic DSP gates
        ↓
Unload TTS and release VRAM
        ↓
Qwen2.5-Omni-7B GPTQ Int4
        ↓
Review each short segment independently
        ↓
Unload reviewer
        ↓
Regenerate rejected segments only
        ↓
FFmpeg render and private-first YouTube upload
        ↓
Analytics replay and subscriber-aware Thompson sampling
```

The models are deliberately serialized. The script model, TTS model and reviewer are never expected to remain resident together on the T4.

## Cost guard

The Modal production contract is intentionally bounded:

- One scheduled run per day.
- One T4 container maximum.
- Thirty-minute function timeout.
- One infrastructure retry maximum.
- Persistent model cache prevents repeated downloads.
- Private uploads during validation.
- Set the Modal workspace budget to **$30/month** before deployment.

At the published T4 rate, Modal's $30 monthly compute credit represents about 50.8 T4 hours. A daily job consuming ten minutes averages five T4 hours per month. Even a full thirty-minute allocation every day is fifteen T4 hours before a retry. The workspace budget is still mandatory because storage, CPU, manual canaries and unexpected retries also consume credit.

This is a free-credit proof environment, not a guarantee that every possible workload remains permanently free. The controller fails closed; it does not silently move to a paid GPU.

## Open-weight audio review

The default reviewer is:

```text
Qwen/Qwen2.5-Omni-7B-GPTQ-Int4
```

It processes each narration segment separately and returns text-only JSON. Segment review keeps the memory envelope practical on a 16 GB T4 and maps directly to selective repair. The talker is disabled because the reviewer never generates speech.

`GPT-Realtime-2.1` remains an optional later upgrade through:

```env
REVIEWER_BACKEND=openai
OPENAI_API_KEY=...
```

No OpenAI key is required for the proof-stage deployment.

## Modal activation

### 1. Install and authenticate

Linux/macOS:

```bash
bash cloud/deploy_modal.sh
```

Windows PowerShell:

```powershell
.\cloud\deploy_modal.ps1
```

The script installs the Modal CLI and opens Modal's account authorization flow.

### 2. Create the owner secret

```bash
modal secret create ai-growth-factory-secrets \
  YOUTUBE_OAUTH_JSON='{"client_id":"...","client_secret":"...","refresh_token":"..."}' \
  PUBLISH_ENABLED=true \
  YOUTUBE_PRIVACY_STATUS=private
```

Do not put OAuth credentials in the repository or image.

### 3. Set the hard workspace budget

In Modal workspace settings, set the monthly budget to **$30** before deploying. This prevents proof-stage experimentation from becoming an unplanned bill.

### 4. Deploy the schedule

```bash
modal deploy cloud/modal_app.py
```

The deployed function runs daily at 10:00 in `Africa/Cairo`.

### 5. Run a private canary

```bash
modal run cloud/modal_app.py --canary
```

Keep the first three uploads private. Review the generated video, voice-review manifest, sources and YouTube metadata before moving to unlisted or public publication.

## Lightning AI fallback

Upload the repository to a free Lightning Studio, then run:

```bash
bash cloud/lightning_bootstrap.sh
```

Lightning is used for:

- Model compatibility tests.
- Prompt and voice-contract experiments.
- Diagnosing failed Modal runs.
- Manual private canaries.

It is not the daily production scheduler because free Studio sessions restart and available GPU credits can vary.

## Promotion gates

The pipeline may move from private to unlisted only after three consecutive canaries satisfy all of these:

- Primary-source validation passes.
- Narration passes objective DSP checks.
- Open-weight reviewer passes local thresholds.
- No unexplained visual, audio or metadata defect.
- No copyright or policy warning.

Public publication remains a separate owner decision. Paid infrastructure becomes justified only when measured channel or conversion revenue covers the projected monthly compute cost with a safety margin.

## Local validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q factory api tests cloud
python -m factory doctor
```

## Important files

- `cloud/modal_app.py` — scheduled T4 production worker.
- `cloud/deploy_modal.sh` / `.ps1` — Modal bootstrap.
- `cloud/lightning_bootstrap.sh` — Lightning development setup.
- `factory/qwen_omni_reviewer.py` — open-weight segment reviewer.
- `factory/voice_pipeline.py` — bounded review and selective repair loop.
- `.env.example` — provider-independent configuration.
- `DEPLOYMENT_STATUS.md` — exact deployed and undeployed boundaries.
- `QUALITY_REPORT.md` — test evidence and limitations.

## Version 1.3.1 corrective media gate

The 1.3.0 media files are invalidated. They used eSpeak timing audio and a quantized FFmpeg `zoompan` effect that produced hold-jump stutter. They must not be used for publication or quality approval.

Version 1.3.1 adds:

- pixel-stable editorial scenes with semantic cuts;
- consecutive-frame temporal analysis;
- explicit rejection of hold-jump motion;
- Qwen3-TTS generator provenance in the voice manifest;
- mandatory perceptual-review approval and threshold checks;
- fail-closed rejection of eSpeak, missing provenance, or missing reviews;
- scene-specific source indices so publisher labels match the supporting source;
- a renamed mechanical `generate_visual_fixture.py` script that requires explicit opt-in and writes `NOT_A_PRODUCTION_CANARY.txt`.

No replacement production canary is included. A production canary is valid only after the real Modal T4 worker generates Qwen3-TTS audio, Qwen2.5-Omni approves it, temporal QC passes, and the resulting upload remains private for owner review.

See `EXPERT_AUDIT_1.3.0.md` for the full root-cause analysis.

## Repository automation

This repository includes three GitHub Actions workflows:

- `CI` — Python 3.12/3.13 tests, branch coverage, compilation and credential/media preflight.
- `Deploy Modal GPU Worker` — manual, protected deployment of the scheduled T4 application.
- `Run Private GPU Canary` — manual invocation of the real-model private canary.

See [`docs/GPU_DEPLOYMENT.md`](docs/GPU_DEPLOYMENT.md) for the exact secret and promotion contract.
