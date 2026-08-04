# v28 Beat-Aligned Production Verification

This branch is accepted only when the exact pre-merge Modal artifact proves all of the following:

- 138–146 WPM narration, with 142 WPM as the default target.
- No tempo correction above 1.15×.
- Grammatical narration with no pasted source-title fragments.
- 14–20 duration-adaptive editorial shots.
- At least four unique shots in the first ten seconds.
- No shot longer than 4.25 seconds.
- Every shot is identified by beat, narration segment, package scene, and exact spoken claim.
- Every executable visual prompt contains the spoken claim active during that shot.
- Exactly one unique keyframe and one unique media asset per shot.
- No source-video looping and no reused media hashes.
- Full-frame visuals with no destructive lower-third matte.
- Three Wan hero shots; all other image shots receive deterministic camera motion.
- Phrase-level captions with at most two one-word cues and no cue shorter than 0.65 seconds.
- 1080×1920 H.264/AAC output at 30 fps.
- CPU tests pass on Python 3.12 and 3.13.
- The downloaded MP4 is manually reviewed before merge.

Publishing remains disabled. A green workflow is necessary but not sufficient; the artifact must also pass visual and audio inspection.
