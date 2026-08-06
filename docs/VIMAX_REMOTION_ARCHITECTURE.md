# ViMax + Remotion production architecture

## Authority boundaries

- **AI Growth Factory** retains source collection, factual grounding, narration, TTS, media inference, review, QC, packaging, and publishing safety.
- **ViMax Script2Video** becomes the optional storyboard authority. Only its text-planning stages are used: characters, storyboard, shot decomposition, first/last frames, motion, and camera tree.
- **Remotion** becomes the optional final timeline renderer for camera-aware still motion, generated clips, phrase captions, source badges, narration, and music.
- **FLUX/SDXL, Wan2.2, Qwen TTS, and Qwen Omni** remain the open-weight media and review backends.

## Feature gates

Both replacements are disabled by default so the existing production canary remains an A/B baseline:

```bash
VIMAX_PLANNER_ENABLED=false
VIDEO_RENDER_BACKEND=ffmpeg
```

Enable ViMax only after a checkout of HKUDS/ViMax at commit
`05a48943878312d88fe5a016c12a9654940ecc43` is installed in an isolated Python environment.
Enable Remotion only after the exact renderer dependencies are installed and `npm run build` has produced `renderer/dist/render.js`.

```bash
VIMAX_PLANNER_ENABLED=true
VIMAX_ROOT=/opt/vimax
VIMAX_PYTHON=/opt/vimax/.venv/bin/python
VIDEO_RENDER_BACKEND=remotion
REMOTION_RENDERER_DIR=/opt/ai-growth-factory/renderer
```

An enabled adapter never silently falls back. Missing dependencies, invalid ViMax artifacts,
non-contiguous frames, missing assets, duration drift, and renderer failures stop the run before publishing.

## Data flow

```text
source-grounded VideoPackage
  -> reviewed narration script
  -> ViMax planning-only subprocess
  -> VisualPlan + factory ShotSpec timeline
  -> existing keyframe and Wan generation/review
  -> vimax-remotion-v1 render spec
  -> isolated Remotion public asset stage
  -> deterministic MP4 + digest-bound manifests
  -> existing video QC and manual canary review
```

## Rollout

1. Merge contract, renderer, adapters, and tests with both flags disabled.
2. Run the existing FFmpeg canary as the baseline.
3. Enable Remotion only and review its exact MP4 artifact.
4. Enable ViMax planning plus Remotion and review the exact MP4 artifact.
5. Remove the obsolete storyboard and FFmpeg compatibility layers only after the new path repeatedly passes.

Publishing remains disabled throughout migration.
