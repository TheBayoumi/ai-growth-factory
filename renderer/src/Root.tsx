import React from 'react';
import type {CalculateMetadataFunction} from 'remotion';
import {Composition} from 'remotion';
import {ShortVideo} from './ShortVideo.js';
import type {RenderSpec} from './schema.js';
import {parseRenderSpec} from './schema.js';

const defaultProps: RenderSpec = {
  schema_version: 'vimax-remotion-v1',
  width: 1080,
  height: 1920,
  fps: 30,
  duration_in_frames: 30,
  audio_path: 'assets/narration.wav',
  background_music_path: null,
  title: '',
  source_label: '',
  transition_frames: 5,
  shots: [
    {
      shot_id: 0,
      start_frame: 0,
      duration_in_frames: 30,
      semantic_claim: 'Preview placeholder',
      purpose: 'preview',
      first_frame_prompt: 'Preview placeholder',
      last_frame_prompt: '',
      motion_prompt: 'Static camera',
      camera: {shot_size: 'medium', angle: 'eye_level', movement: 'static'},
      renderer: 'image_motion',
      media_path: 'assets/preview.png',
      keyframe_path: null,
      source_index: 0,
      seed: 0,
      reference_assets: [],
    },
  ],
  captions: [],
};

const calculateMetadata: CalculateMetadataFunction<RenderSpec> = ({props}) => {
  const spec = parseRenderSpec(props);
  return {
    durationInFrames: spec.duration_in_frames,
    width: spec.width,
    height: spec.height,
    fps: spec.fps,
    props: spec,
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="ShortVideo"
      component={ShortVideo}
      durationInFrames={defaultProps.duration_in_frames}
      fps={defaultProps.fps}
      width={defaultProps.width}
      height={defaultProps.height}
      defaultProps={defaultProps}
      calculateMetadata={calculateMetadata}
    />
  );
};
