import type {RenderShot} from './schema.js';

export type TransitionBoundary = {
  outgoing_shot_id: number;
  incoming_shot_id: number;
  start_frame: number;
  duration_in_frames: number;
  transition: 'opacity_crossfade';
};

export const transitionFramesBetween = (
  outgoing: RenderShot,
  incoming: RenderShot,
  requestedFrames: number,
): number => {
  if (outgoing.renderer !== 'image_motion' || requestedFrames <= 0) {
    return 0;
  }
  return Math.max(
    0,
    Math.min(
      requestedFrames,
      Math.floor(outgoing.duration_in_frames / 3),
      Math.floor(incoming.duration_in_frames / 3),
    ),
  );
};

export const buildTransitionPlan = (
  shots: RenderShot[],
  requestedFrames: number,
): TransitionBoundary[] => {
  const transitions: TransitionBoundary[] = [];
  for (let index = 0; index + 1 < shots.length; index += 1) {
    // The loop condition proves both entries exist; explicit assertions preserve that fact
    // under TypeScript's noUncheckedIndexedAccess mode.
    const outgoing = shots[index]!;
    const incoming = shots[index + 1]!;
    const frames = transitionFramesBetween(outgoing, incoming, requestedFrames);
    if (frames <= 0) {
      continue;
    }
    transitions.push({
      outgoing_shot_id: outgoing.shot_id,
      incoming_shot_id: incoming.shot_id,
      start_frame: incoming.start_frame,
      duration_in_frames: frames,
      transition: 'opacity_crossfade',
    });
  }
  return transitions;
};
