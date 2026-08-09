import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Html5Audio,
  Img,
  interpolate,
  OffthreadVideo,
  Sequence,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import type {CaptionCue, RenderShot, RenderSpec} from './schema.js';
import {transitionFramesBetween} from './transitions.js';

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const asset = (value: string): string => staticFile(value.replace(/^\/+/, ''));

const imageTransform = (
  movement: string,
  progress: number,
): {scale: number; x: number; y: number} => {
  const eased = Easing.inOut(Easing.cubic)(progress);
  switch (movement) {
    case 'dolly_out':
      return {scale: 1.085 - 0.075 * eased, x: 0, y: 0};
    case 'pan_left':
      return {scale: 1.065, x: 2.8 - 5.6 * eased, y: 0};
    case 'pan_right':
      return {scale: 1.065, x: -2.8 + 5.6 * eased, y: 0};
    case 'tilt_up':
      return {scale: 1.06, x: 0, y: 2.4 - 4.8 * eased};
    case 'tilt_down':
      return {scale: 1.06, x: 0, y: -2.4 + 4.8 * eased};
    case 'static':
      return {scale: 1.012 + 0.018 * eased, x: 0, y: 0};
    case 'dolly_in':
    case 'subtle_push_in':
    default:
      return {scale: 1.0 + 0.075 * eased, x: 0, y: 0};
  }
};

const ImageShot: React.FC<{shot: RenderShot; incomingFrames: number}> = ({
  shot,
  incomingFrames,
}) => {
  const frame = useCurrentFrame();
  const motionFrame = Math.min(frame, Math.max(0, shot.duration_in_frames - 1));
  const progress = interpolate(
    motionFrame,
    [0, Math.max(1, shot.duration_in_frames - 1)],
    [0, 1],
    clamp,
  );
  const transform = imageTransform(shot.camera.movement, progress);
  const opacity =
    incomingFrames > 0
      ? interpolate(frame, [0, incomingFrames], [0, 1], clamp)
      : 1;

  return (
    <AbsoluteFill style={{backgroundColor: '#05070c', opacity, overflow: 'hidden'}}>
      <Img
        src={asset(shot.media_path)}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: `translate3d(${transform.x}%, ${transform.y}%, 0) scale(${transform.scale})`,
          transformOrigin: 'center center',
        }}
      />
      <AbsoluteFill
        style={{
          background:
            'linear-gradient(180deg, rgba(3,5,10,0.08) 0%, rgba(3,5,10,0) 58%, rgba(3,5,10,0.24) 100%)',
        }}
      />
    </AbsoluteFill>
  );
};

const VideoShot: React.FC<{shot: RenderShot; incomingFrames: number}> = ({
  shot,
  incomingFrames,
}) => {
  const frame = useCurrentFrame();
  const opacity =
    incomingFrames > 0
      ? interpolate(frame, [0, incomingFrames], [0, 1], clamp)
      : 1;
  return (
    <AbsoluteFill style={{backgroundColor: '#05070c', opacity, overflow: 'hidden'}}>
      <OffthreadVideo
        src={asset(shot.media_path)}
        muted
        style={{width: '100%', height: '100%', objectFit: 'cover'}}
      />
      <AbsoluteFill
        style={{
          background:
            'linear-gradient(180deg, rgba(3,5,10,0.06) 0%, rgba(3,5,10,0) 62%, rgba(3,5,10,0.20) 100%)',
        }}
      />
    </AbsoluteFill>
  );
};

const CaptionCard: React.FC<{cue: CaptionCue; width: number; height: number}> = ({
  cue,
  width,
  height,
}) => {
  const frame = useCurrentFrame();
  const duration = cue.end_frame - cue.start_frame;
  const intro = Math.min(7, Math.max(3, Math.floor(duration * 0.18)));
  const outro = Math.min(5, Math.max(2, Math.floor(duration * 0.12)));
  const opacityIn = interpolate(frame, [0, intro], [0, 1], clamp);
  const opacityOut = interpolate(
    frame,
    [Math.max(0, duration - outro), duration],
    [1, 0],
    clamp,
  );
  const scale = interpolate(frame, [0, intro], [0.965, 1], clamp);
  const fontSize = Math.round(width * (cue.text.length > 30 ? 0.052 : 0.061));

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: Math.round(height * 0.105),
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          maxWidth: Math.round(width * 0.88),
          padding: `${Math.round(width * 0.018)}px ${Math.round(width * 0.034)}px`,
          borderRadius: Math.round(width * 0.028),
          color: '#ffffff',
          background: 'rgba(5, 8, 15, 0.78)',
          border: `${Math.max(1, Math.round(width / 540))}px solid rgba(255,255,255,0.18)`,
          boxShadow: `0 ${Math.round(width * 0.014)}px ${Math.round(width * 0.05)}px rgba(0,0,0,0.38)`,
          fontFamily: 'Arial, Helvetica, sans-serif',
          fontWeight: 800,
          fontSize,
          lineHeight: 1.08,
          textAlign: 'center',
          letterSpacing: '-0.025em',
          textWrap: 'balance',
          opacity: Math.min(opacityIn, opacityOut),
          transform: `scale(${scale})`,
        }}
      >
        {cue.text}
      </div>
    </AbsoluteFill>
  );
};

const CaptionTrack: React.FC<{spec: RenderSpec}> = ({spec}) => {
  return (
    <>
      {spec.captions.map((cue) => (
        <Sequence
          key={cue.cue_id}
          from={cue.start_frame}
          durationInFrames={cue.end_frame - cue.start_frame}
          premountFor={Math.min(spec.fps, 15)}
        >
          <CaptionCard cue={cue} width={spec.width} height={spec.height} />
        </Sequence>
      ))}
    </>
  );
};

const SourceBadge: React.FC<{spec: RenderSpec}> = ({spec}) => {
  const frame = useCurrentFrame();
  if (!spec.source_label) {
    return null;
  }
  const visibleAtStart = interpolate(
    frame,
    [0, 8, spec.fps * 3, spec.fps * 3.4],
    [0, 1, 1, 0],
    clamp,
  );
  const endStart = Math.max(0, spec.duration_in_frames - Math.round(spec.fps * 2.5));
  const visibleAtEnd = interpolate(
    frame,
    [endStart, endStart + 8, spec.duration_in_frames - 5, spec.duration_in_frames],
    [0, 1, 1, 0],
    clamp,
  );
  const opacity = Math.max(visibleAtStart, visibleAtEnd);
  return (
    <AbsoluteFill
      style={{
        alignItems: 'flex-start',
        padding: Math.round(spec.width * 0.045),
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          opacity,
          maxWidth: Math.round(spec.width * 0.80),
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          padding: `${Math.round(spec.width * 0.012)}px ${Math.round(spec.width * 0.022)}px`,
          borderRadius: 999,
          background: 'rgba(5,8,15,0.66)',
          border: '1px solid rgba(255,255,255,0.16)',
          color: 'rgba(255,255,255,0.86)',
          fontFamily: 'Arial, Helvetica, sans-serif',
          fontSize: Math.round(spec.width * 0.026),
          fontWeight: 700,
          letterSpacing: '0.01em',
        }}
      >
        Sources: {spec.source_label}
      </div>
    </AbsoluteFill>
  );
};

export const ShortVideo: React.FC<RenderSpec> = (spec) => {
  return (
    <AbsoluteFill style={{backgroundColor: '#05070c'}}>
      {spec.shots.map((shot, index) => {
        const previous = index > 0 ? spec.shots[index - 1] : undefined;
        const next = index + 1 < spec.shots.length ? spec.shots[index + 1] : undefined;
        const incomingFrames = previous
          ? transitionFramesBetween(previous, shot, spec.transition_frames)
          : 0;
        const outgoingFrames = next
          ? transitionFramesBetween(shot, next, spec.transition_frames)
          : 0;
        return (
          <Sequence
            key={shot.shot_id}
            from={shot.start_frame}
            durationInFrames={shot.duration_in_frames + outgoingFrames}
            premountFor={Math.min(spec.fps, 15)}
            style={{zIndex: shot.shot_id + 1}}
          >
            {shot.renderer === 'video_clip' ? (
              <VideoShot shot={shot} incomingFrames={incomingFrames} />
            ) : (
              <ImageShot shot={shot} incomingFrames={incomingFrames} />
            )}
          </Sequence>
        );
      })}

      <AbsoluteFill style={{zIndex: 1000}}>
        <CaptionTrack spec={spec} />
        <SourceBadge spec={spec} />
      </AbsoluteFill>

      <Html5Audio src={asset(spec.audio_path)} volume={1} />
      {spec.background_music_path ? (
        <Html5Audio src={asset(spec.background_music_path)} volume={0.72} />
      ) : null}
    </AbsoluteFill>
  );
};
