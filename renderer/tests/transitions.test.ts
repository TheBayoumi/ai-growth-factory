import {describe, expect, it} from 'vitest';
import type {RenderShot} from '../src/schema.js';
import {buildTransitionPlan, transitionFramesBetween} from '../src/transitions.js';

const shot = (
  shotId: number,
  renderer: RenderShot['renderer'],
  startFrame: number,
  durationInFrames = 90,
): RenderShot => ({
  shot_id: shotId,
  start_frame: startFrame,
  duration_in_frames: durationInFrames,
  semantic_claim: `claim ${shotId}`,
  purpose: 'support_claim',
  first_frame_prompt: `first ${shotId}`,
  last_frame_prompt: `last ${shotId}`,
  motion_prompt: 'subtle push in',
  camera: {shot_size: 'medium', angle: 'eye_level', movement: 'subtle_push_in'},
  renderer,
  media_path: `shot-${shotId}.png`,
  keyframe_path: `shot-${shotId}.png`,
  source_index: 0,
  seed: shotId + 1,
  reference_assets: [],
});

describe('Remotion transition policy', () => {
  it('realizes symmetric opacity crossfades after image-motion shots', () => {
    const shots = [
      shot(0, 'image_motion', 0),
      shot(1, 'image_motion', 90),
      shot(2, 'video_clip', 180),
      shot(3, 'image_motion', 270),
      shot(4, 'image_motion', 360),
    ];

    const transitions = buildTransitionPlan(shots, 5);

    expect(transitions).toEqual([
      {
        outgoing_shot_id: 0,
        incoming_shot_id: 1,
        start_frame: 90,
        duration_in_frames: 5,
        transition: 'opacity_crossfade',
      },
      {
        outgoing_shot_id: 1,
        incoming_shot_id: 2,
        start_frame: 180,
        duration_in_frames: 5,
        transition: 'opacity_crossfade',
      },
      {
        outgoing_shot_id: 3,
        incoming_shot_id: 4,
        start_frame: 360,
        duration_in_frames: 5,
        transition: 'opacity_crossfade',
      },
    ]);
  });

  it('keeps video-clip exits as intentional hard cuts', () => {
    const outgoing = shot(0, 'video_clip', 0);
    const incoming = shot(1, 'image_motion', 90);

    expect(transitionFramesBetween(outgoing, incoming, 5)).toBe(0);
  });

  it('clamps both sequence overlap and incoming fade to the same safe duration', () => {
    const outgoing = shot(0, 'image_motion', 0, 9);
    const incoming = shot(1, 'image_motion', 9, 6);

    expect(transitionFramesBetween(outgoing, incoming, 5)).toBe(2);
  });
});
