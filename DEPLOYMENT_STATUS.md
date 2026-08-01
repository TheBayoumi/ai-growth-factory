# Deployment status — version 1.3.1

## Corrective release state

Version 1.3.1 is built and tested locally. It is **not deployed** to Modal and has not produced a real Qwen3-TTS canary.

## Live control plane

- Alias: `https://ai-growth-factory.vercel.app`
- Last independently verified live release: version `1.2.0`
- Deployment: `dpl_F2QWaBbMaCesTgx4jQFqonHJpZpA`
- Purpose: status page and YouTube OAuth bootstrap only.
- GPU/model execution: disabled by design.

The package contains updated 1.3.1 control-plane source, but this corrective turn did not redeploy it.

## Modal worker

Implemented but not deployed because no authorized Modal workspace or owner YouTube secret is connected. Activation still requires:

1. `modal setup`;
2. a $30 workspace budget cap;
3. the owner `YOUTUBE_OAUTH_JSON` secret;
4. `modal deploy cloud/modal_app.py`;
5. a real private canary with Qwen3-TTS and Qwen2.5-Omni.

## Publication state

Publishing must remain disabled until all of the following pass on the same artifact:

- Qwen3-TTS provenance;
- approved perceptual-review scores;
- deterministic audio QC;
- zero hold-jump temporal-stutter windows;
- correct scene-to-source mapping;
- private owner review.
