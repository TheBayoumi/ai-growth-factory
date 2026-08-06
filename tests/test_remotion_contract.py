from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.remotion_bridge import render_with_remotion, stage_render_assets
from factory.remotion_contract import (
    RemotionContractError,
    build_remotion_render_spec,
)


class RemotionContractTests(unittest.TestCase):
    def _fixture(self, root: Path):
        audio = root / "narration.wav"
        music = root / "music.wav"
        audio.write_bytes(b"RIFF" + b"a" * 100)
        music.write_bytes(b"RIFF" + b"b" * 100)
        shots = []
        media = []
        starts = [0.0, 2.0, 4.0]
        for index, start in enumerate(starts):
            path = root / f"media-{index}.png"
            keyframe = root / f"keyframe-{index}.png"
            path.write_bytes(b"png" + bytes([index]) * 100)
            keyframe.write_bytes(b"key" + bytes([index]) * 100)
            shots.append(
                SimpleNamespace(
                    shot_id=index,
                    start_seconds=start,
                    semantic_claim=f"claim {index}",
                    treatment="wide establishing view" if index == 0 else "tight detail",
                    visual_direction="slow dolly in",
                    source_index=0,
                    seed=index,
                )
            )
            media.append(
                SimpleNamespace(
                    scene_index=index,
                    media_type="image",
                    path=path,
                    keyframe_path=keyframe,
                    director_prompt=f"prompt {index}",
                    prompt=f"compiled {index}",
                )
            )
        segments = [
            SimpleNamespace(segment_id=0, start_seconds=0.0, end_seconds=3.0, text="First phrase"),
            SimpleNamespace(segment_id=1, start_seconds=3.0, end_seconds=6.0, text="Second phrase"),
        ]
        cues = [
            SimpleNamespace(start_seconds=0.0, end_seconds=1.5, text="First"),
            SimpleNamespace(start_seconds=1.5, end_seconds=3.0, text="phrase"),
            SimpleNamespace(start_seconds=3.0, end_seconds=6.0, text="Second phrase"),
        ]
        package = SimpleNamespace(title="Test", source_publishers=["Source A", "Source A"])
        spec = build_remotion_render_spec(
            shots=shots,
            media=media,
            segments=segments,
            caption_cues=cues,
            package=package,
            audio_path=audio,
            background_music_path=music,
            width=1080,
            height=1920,
            fps=30,
            duration_seconds=6.0,
        )
        return spec

    def test_builds_frame_authoritative_phrase_caption_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            spec = self._fixture(Path(temp))
            self.assertEqual(180, spec.duration_in_frames)
            self.assertEqual([0, 60, 120], [item.start_frame for item in spec.shots])
            self.assertEqual([60, 60, 60], [item.duration_in_frames for item in spec.shots])
            self.assertEqual(["First", "phrase", "Second phrase"], [item.text for item in spec.captions])
            self.assertEqual("Source A", spec.source_label)
            spec.validate(require_files=True)

    def test_stages_all_assets_under_isolated_public_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self._fixture(root)
            staged, path = stage_render_assets(spec=spec, stage_root=root / "stage")
            self.assertTrue(path.is_file())
            self.assertEqual("assets/narration.wav", staged.audio_path)
            self.assertEqual("assets/background-music.wav", staged.background_music_path)
            for shot in staged.shots:
                self.assertFalse(Path(shot.media_path).is_absolute())
                self.assertTrue((root / "stage" / "public" / shot.media_path).is_file())

    def test_render_manifest_binds_to_persisted_staged_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer"
            (renderer / "dist").mkdir(parents=True)
            (renderer / "package.json").write_text("{}", encoding="utf-8")
            (renderer / "dist" / "render.js").write_text("", encoding="utf-8")
            workdir = root / "work"
            output = workdir / "video.mp4"
            spec = self._fixture(root)

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"video" * 120_000)
                return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

            with patch.dict(os.environ, {"REMOTION_RENDERER_DIR": str(renderer)}), patch(
                "factory.remotion_bridge.subprocess.run",
                side_effect=fake_run,
            ):
                _output, manifest_path, _log = render_with_remotion(
                    spec=spec,
                    output_path=output,
                    workdir=workdir,
                )

            persisted = workdir / "remotion-staged-render-spec.json"
            self.assertTrue(persisted.is_file())
            staged_payload = json.loads(persisted.read_text(encoding="utf-8"))
            self.assertEqual("assets/narration.wav", staged_payload["audio_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(str(persisted), manifest["staged_render_spec"])
            self.assertTrue(Path(manifest["staged_render_spec"]).is_file())

    def test_rejects_noncontiguous_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            spec = self._fixture(Path(temp))
            broken = type(spec)(**{**spec.__dict__, "shots": (spec.shots[0], spec.shots[2])})
            with self.assertRaises(RemotionContractError):
                broken.validate()


if __name__ == "__main__":
    unittest.main()
