# Live-trend autonomous production verification v14

This merge requests one CI-gated Modal A10 production run from source proven on Python 3.12 and 3.13.

Acceptance requires:

- current official evidence and deterministic URL-to-publisher attribution
- current AI trend signals from multiple providers
- reviewed Qwen3-TTS narration within the unchanged bounded recovery loop
- 0.5 dB normalization safety headroom while preserving the -1.00 dBFS acceptance gate
- a repaired final voice attempt must reach Omni when deterministic QC passes
- deterministic final QC must override stale prior reviewer messaging
- six semantically distinct, text-free generated keyframes
- exactly three valid Wan2.2 hero clips
- separate animated ASS captions
- dedicated platform thumbnail output
- 1080x1920 final video at the 30 fps target
- complete trend, package, voice, prompt, seed, media, composition, and QC manifests

Publishing remains disabled until the downloaded video and narration pass perceptual inspection.
