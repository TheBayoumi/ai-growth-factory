# Quality report — version 1.3.1

## Corrective scope

Version 1.3.1 is a corrective release following a perceptual audit of the 1.3.0 media outputs. The 1.3.0 videos are invalidated and must not be used for publication decisions.

## Validation completed

- 45 automated tests passed.
- Branch-aware project coverage: 70%.
- Stable FFmpeg renderer test passed.
- Temporal hold-jump detector unit tests passed.
- All three 1.3.0 videos are rejected by the new temporal gate.
- Stable dashboard render fixture reports zero stutter windows and zero jump ratio in all six sampled scene windows.
- Missing production voice manifest fails closed.
- eSpeak provenance fails closed.
- Missing perceptual review fails closed.
- Valid Qwen3-TTS provenance plus approved reviewer scores passes the publication-voice gate.
- Scene source indices are range-checked and mapped to the correct publisher.

## Stable fixture result

The internal dashboard visual fixture produced these temporal results:

- temporal stutter windows: `0/6`;
- jump ratio: `0.0` in every scene window;
- maximum within-scene frame difference: `0.0011` on a 0–255 luminance scale.

This verifies removal of the quantized zoompan defect. The fixture contains mechanical eSpeak timing audio and is intentionally not distributed as a production canary.

## Not yet verified

The following require an authenticated Modal T4 or equivalent GPU environment:

- real Qwen3-TTS narration quality;
- Qwen2.5-Omni perceptual review on the generated waveform;
- actual GPU memory compatibility and runtime;
- final mixed-audio noise floor and naturalness;
- a complete private YouTube canary.

The pipeline remains fail-closed until those checks pass.
