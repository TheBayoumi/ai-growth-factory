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
```

It is not part of the zero-API-cost proof stage.

## Repository workflows

- `.github/workflows/ci.yml` runs verification only after changes land on `main`.
- `.github/workflows/modal-deploy.yml` deploys the GPU worker manually from a protected environment.
- `.github/workflows/modal-canary.yml` runs a private GPU canary manually.

## Safety contract

A video cannot reach the publisher unless all of the following pass:

1. Primary-source freshness and evidence checks.
2. Qwen3-TTS provenance in the narration manifest.
3. Deterministic audio checks.
4. Approved perceptual review.
5. Stable video motion and codec checks.
6. Correct source mapping.
7. Private-first publishing policy during validation.

Placeholder narration, eSpeak output, missing review evidence, hold-jump motion and stale sources all fail closed.

## Local verification

```bash
python -m pip install -e . coverage
python -m compileall -q factory api tests cloud
coverage run --branch -m unittest discover -s tests -v
coverage report --fail-under=65
python scripts/repository_preflight.py
```

## Modal deployment

Required GitHub environment: `modal-production`

Required repository or environment secrets:

```text
MODAL_TOKEN_ID
MODAL_TOKEN_SECRET
```

The Modal worker also expects a Modal secret named `ai-growth-factory-secrets` containing the YouTube OAuth JSON and private-first publishing controls.

Deploy through GitHub Actions using **Deploy Modal GPU Worker**, or locally:

```bash
pip install "modal==1.5.3"
modal deploy cloud/modal_app.py
```

Run a private canary:

```bash
modal run cloud/modal_app.py --canary
```

## Current status

- Source and tests: ready.
- Vercel control plane: deployed separately.
- GitHub repository: initialized.
- Modal GPU worker: requires authorized Modal tokens and YouTube OAuth credentials.
- Public publishing: intentionally disabled until private canaries pass perceptual review.
