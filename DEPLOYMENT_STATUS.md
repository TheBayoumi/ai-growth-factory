# Deployment status — Modal only

## Deployment policy

AI Growth Factory is a **Modal-only production runtime**.

Vercel is not an approved deployment target for this repository. Automatic Git deployments to Vercel are disabled by the repository-level `vercel.json` policy:

```json
{
  "git": {
    "deploymentEnabled": false
  }
}
```

Do not deploy this repository, any branch, preview, control plane, OAuth bootstrap, status page, renderer, API, or production worker to Vercel.

## Production runtime

- Production compute: Modal.
- GPU/model execution: Modal only.
- ViMax planning, Qwen TTS/review, image/video inference, Remotion rendering, persistent canary artifacts, and production scheduling belong to the Modal execution path.
- GitHub Actions is used only for CI, validation, orchestration, and protected Modal invocation/deployment.
- Publishing remains fail-closed until the exact production artifact passes all required automated and manual review gates.

## Vercel state

The legacy Vercel control-plane architecture is retired. The currently connected Vercel account exposes no `ai-growth-factory` project, and repository Git deployments to Vercel are explicitly disabled. Any future architecture or automation change that reintroduces Vercel deployment is out of policy.

## Modal deployment

Use only the repository's protected Modal workflows or the documented Modal CLI path for deployment and canary execution. Production secrets must remain in the protected `modal-production` environment / Modal secret store and must never be committed to the repository.
