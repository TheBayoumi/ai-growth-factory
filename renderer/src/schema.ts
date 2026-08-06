import {z} from 'zod';

export const CameraSpecSchema = z.object({
  shot_size: z.string().min(1),
  angle: z.string().min(1),
  movement: z.string().min(1),
});

export const CaptionCueSchema = z.object({
  cue_id: z.number().int().nonnegative(),
  start_frame: z.number().int().nonnegative(),
  end_frame: z.number().int().positive(),
  text: z.string().min(1),
});

export const RenderShotSchema = z.object({
  shot_id: z.number().int().nonnegative(),
  start_frame: z.number().int().nonnegative(),
  duration_in_frames: z.number().int().positive(),
  semantic_claim: z.string().min(1),
  purpose: z.string().min(1),
  first_frame_prompt: z.string().min(1),
  last_frame_prompt: z.string(),
  motion_prompt: z.string().min(1),
  camera: CameraSpecSchema,
  renderer: z.enum(['image_motion', 'video_clip']),
  media_path: z.string().min(1),
  keyframe_path: z.string().nullable(),
  source_index: z.number().int().nonnegative(),
  seed: z.number().int(),
  reference_assets: z.array(z.string()),
});

export const RenderSpecSchema = z.object({
  schema_version: z.literal('vimax-remotion-v1'),
  width: z.number().int().min(360),
  height: z.number().int().min(640),
  fps: z.number().int().min(12).max(60),
  duration_in_frames: z.number().int().positive(),
  audio_path: z.string().min(1),
  background_music_path: z.string().nullable(),
  title: z.string(),
  source_label: z.string(),
  shots: z.array(RenderShotSchema).min(1),
  captions: z.array(CaptionCueSchema),
  transition_frames: z.number().int().nonnegative(),
});

export type RenderSpec = z.infer<typeof RenderSpecSchema>;
export type RenderShot = z.infer<typeof RenderShotSchema>;
export type CaptionCue = z.infer<typeof CaptionCueSchema>;

export const validateTimeline = (spec: RenderSpec): RenderSpec => {
  let expectedFrame = 0;
  spec.shots.forEach((shot, index) => {
    if (shot.shot_id !== index) {
      throw new Error(`shot IDs must be contiguous: expected ${index}, got ${shot.shot_id}`);
    }
    if (shot.start_frame !== expectedFrame) {
      throw new Error(
        `shot ${shot.shot_id} starts at ${shot.start_frame}; expected ${expectedFrame}`,
      );
    }
    expectedFrame += shot.duration_in_frames;
  });
  if (expectedFrame !== spec.duration_in_frames) {
    throw new Error(
      `shots end at ${expectedFrame}; composition ends at ${spec.duration_in_frames}`,
    );
  }

  spec.captions.forEach((cue, index) => {
    if (cue.cue_id !== index) {
      throw new Error(`caption IDs must be contiguous: expected ${index}, got ${cue.cue_id}`);
    }
    if (cue.start_frame >= cue.end_frame || cue.end_frame > spec.duration_in_frames) {
      throw new Error(`caption ${cue.cue_id} has an invalid frame range`);
    }
  });
  return spec;
};

export const parseRenderSpec = (value: unknown): RenderSpec => {
  return validateTimeline(RenderSpecSchema.parse(value));
};
