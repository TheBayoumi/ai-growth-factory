import {describe, expect, it} from 'vitest';
import {parseRenderSpec} from '../src/schema.js';

const valid = {
  schema_version: 'vimax-remotion-v1',
  width: 1080,
  height: 1920,
  fps: 30,
  duration_in_frames: 60,
  audio_path: 'assets/narration.wav',
  background_music_path: null,
  title: 'Test',
  source_label: 'Publisher',
  transition_frames: 5,
  shots: [
    {
      shot_id: 0,
      start_frame: 0,
      duration_in_frames: 30,
      semantic_claim: 'Claim one',
      purpose: 'hook',
      first_frame_prompt: 'First',
      last_frame_prompt: 'Last',
      motion_prompt: 'Slow dolly in',
      camera: {shot_size: 'wide', angle: 'eye_level', movement: 'dolly_in'},
      renderer: 'image_motion',
      media_path: 'assets/0.png',
      keyframe_path: 'assets/0.png',
      source_index: 0,
      seed: 1,
      reference_assets: [],
    },
    {
      shot_id: 1,
      start_frame: 30,
      duration_in_frames: 30,
      semantic_claim: 'Claim two',
      purpose: 'evidence',
      first_frame_prompt: 'First two',
      last_frame_prompt: 'Last two',
      motion_prompt: 'Static camera',
      camera: {shot_size: 'close_up', angle: 'eye_level', movement: 'static'},
      renderer: 'video_clip',
      media_path: 'assets/1.mp4',
      keyframe_path: 'assets/1.png',
      source_index: 0,
      seed: 2,
      reference_assets: [],
    },
  ],
  captions: [
    {cue_id: 0, start_frame: 0, end_frame: 30, text: 'Caption one'},
    {cue_id: 1, start_frame: 30, end_frame: 60, text: 'Caption two'},
  ],
};

describe('render spec', () => {
  it('accepts a contiguous frame-authoritative timeline', () => {
    expect(parseRenderSpec(valid).duration_in_frames).toBe(60);
  });

  it('rejects timeline gaps', () => {
    const broken = structuredClone(valid);
    broken.shots[1].start_frame = 31;
    expect(() => parseRenderSpec(broken)).toThrow(/expected 30/);
  });

  it('rejects captions outside the composition', () => {
    const broken = structuredClone(valid);
    broken.captions[1].end_frame = 61;
    expect(() => parseRenderSpec(broken)).toThrow(/invalid frame range/);
  });
});
