import type {RenderShot} from './schema.js';

export type TransitionBoundary = {
  outgoing_shot_id: number;
  incoming_shot_id: number;
  start_frame: number;
  duration_in_frames: number;
  transition: 'opacity_crossfade';
};

const packageSceneIndex = (shot: RenderShot): number | null => {
  const match = /(?:^|;)\s*package_scene:(\d+)\b/i.exec(shot.purpose);
  return match ? Number.parseInt(match[1]!, 10) : null;
};

const shouldCrossfade = (outgoing: RenderShot, incoming: RenderShot): boolean => {
  // Preserve the legacy image-exit dissolve for non-ViMax/compatibility renders.
  if (outgoing.renderer === 'image_motion') {
    return true;
  }

  // Real temporal clips use editorial cuts by default. A short opacity dissolve is reserved for
  // a narrative package-scene change, where the story is intentionally moving to a new beat.
  if (outgoing.renderer === 'video_clip' && incoming.renderer === 'video_clip') {
    const outgoingBeat = packageSceneIndex(outgoing);
    const incomingBeat = packageSceneIndex(incoming);
    return outgoingBeat !== null && incomingBeat !== null && outgoingBeat !== incomingBeat;
  }
  return false;
};

export const transitionFramesBetween = (
  outgoing: RenderShot,
  incoming: RenderShot,
  requestedFrames: number,
): number => {
  if (requestedFrames <= 0 || !shouldCrossfade(outgoing, incoming)) {
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
