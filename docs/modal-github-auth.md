# One-time Modal authentication for GitHub Actions

The production workflows must not use `modal token new` inside GitHub Actions. Hosted runners are ephemeral, so the generated `~/.modal.toml` file disappears after each job.

## Create the token

Create a dedicated Modal API token from the Modal workspace token settings, or on a trusted local machine run:

```bash
modal token new
```

Copy the resulting token ID and token secret. Do not commit them and do not paste them into issues, pull requests, workflow logs, or chat.

## Store it once in GitHub

In `TheBayoumi/ai-growth-factory`:

1. Open **Settings → Environments → modal-production**.
2. Under **Environment secrets**, create `MODAL_TOKEN_ID`.
3. Create `MODAL_TOKEN_SECRET`.
4. Keep deployment branch restrictions limited to `main`.

The workflow reads only these encrypted environment secrets. It validates their presence before installing dependencies and validates the credential pair with `modal token info` before deploying.

## Verification

Merge a branch named `verify/modal-gpu-*` into `main`. The production-verification workflow will:

1. Verify the stored Modal credentials.
2. Run the complete test suite.
3. Deploy the T4 worker.
4. Run the real render-only canary with publishing disabled.
5. Download the exact MP4, WAV, package, voice-review manifest, and video-QC report.
6. Fail unless the artifact is complete and the production QC status is `verified_render_canary`.

This setup is performed once. Normal deployments and canary retries do not require browser authorization.
