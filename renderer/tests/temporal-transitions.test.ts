import {describe, expect, it} from 'vitest';
import type {RenderShot} from '../src/schema.js';
import {buildTransitionPlan, transitionFramesBetween} from '../src/transitions.js';

const videoShot = (
  shotId: number,
  startFrame: number,
  packageSceneIndex: number,
  durationInFrames = 90,
): RenderShot => ({
  shot_id: shotId,
  start_frame: startFrame,
  duration_in_frames: durationInFrames,
  semantic_claim: `claim ${shotId}`,
  purpose: `package_scene:${packageSceneIndex}; continuity shot`,
  first_frame_prompt: `first ${shotId}`,
  last_frame_prompt: `last ${shotId}`,
  motion_prompt: 'controlled subject and environment motion',
  camera: {shot_size: 'medium', angle: 'eye_level', movement: 'dolly_in'},
  renderer: 'video_clip',
  media_path: `shot-${shotId}.mp4`,
  keyframe_path: `shot-${shotId}.png`,
  source_index: 0,
  seed: shotId + 1,
  reference_assets: [],
});

describe('all-temporal editorial transition policy', () => {
  it('uses hard cuts inside a story beat and short dissolves only at beat changes', () => {
    const shots = [
      videoShot(0, 0, 0),
      videoShot(1, 90, 0),
      videoShot(2, 180, 1),
      videoShot(3, 270, 1),
      videoShot(4, 360, 2),
    ];

    expect(buildTransitionPlan(shots, 5)).toEqual([
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

  it('does not dissolve adjacent motion clips that belong to the same continuous beat', () => {
    const outgoing = videoShot(0, 0, 3);
    const incoming = videoShot(1, 90, 3);
    expect(transitionFramesBetween(outgoing, incoming, 5)).toBe(0);
  });

  it('clamps a story-beat dissolve to the safe overlap of both temporal clips', () => {
    const outgoing = videoShot(0, 0, 0, 9);
    const incoming = videoShot(1, 9, 1, 6);
    expect(transitionFramesBetween(outgoing, incoming, 5)).toBe(2);
  });
});
