"""动图解码、元信息与抽帧测试。"""

from __future__ import annotations

import unittest

from read_animation.frames import AnimationReader, pick_frame_indices

from tests.image_fixtures import make_animated_gif, make_animated_webp


class PickFrameIndicesTests(unittest.TestCase):
    """采样索引选择。"""

    def test_returns_all_frames_when_request_covers_animation(self) -> None:
        self.assertEqual(pick_frame_indices(8, 8), list(range(8)))
        self.assertEqual(pick_frame_indices(8, 16), list(range(8)))

    def test_samples_evenly_across_timeline(self) -> None:
        indices = pick_frame_indices(24, 8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 23)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(set(indices)), len(indices))

    def test_handles_degenerate_inputs(self) -> None:
        self.assertEqual(pick_frame_indices(0, 4), [])
        self.assertEqual(pick_frame_indices(8, 0), [])
        self.assertEqual(pick_frame_indices(1, 4), [0])


class AnimationReaderTests(unittest.TestCase):
    """GIF/WebP 读取与帧渲染。"""

    def test_reads_gif_metadata(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=8, duration=80), "gif", max_scan_frames=3000)
        try:
            meta = reader.meta
            self.assertEqual(meta.kind, "gif")
            self.assertEqual(meta.frame_count, 8)
            self.assertEqual(meta.total_duration_ms, 640)
            self.assertFalse(meta.scan_truncated)
            self.assertEqual(meta.loop, 0)
            self.assertAlmostEqual(meta.duration_seconds, 0.64, places=3)
            self.assertGreater(meta.fps, 10.0)
            self.assertLess(meta.fps, 20.0)
        finally:
            reader.close()

    def test_reads_webp_metadata(self) -> None:
        reader = AnimationReader(make_animated_webp(frame_count=6, duration=100), "webp", max_scan_frames=3000)
        try:
            meta = reader.meta
            self.assertEqual(meta.kind, "webp")
            self.assertEqual(meta.frame_count, 6)
            self.assertEqual(meta.total_duration_ms, 600)
        finally:
            reader.close()

    def test_scan_truncation_is_reported(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=8), "gif", max_scan_frames=3)
        try:
            self.assertTrue(reader.meta.scan_truncated)
            with self.assertRaises(Exception):
                reader.sample(4)
        finally:
            reader.close()

    def test_samples_carry_timing(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=8, duration=80), "gif", max_scan_frames=3000)
        try:
            samples = reader.sample(4)
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0].index, 0)
            self.assertEqual(samples[0].start_ms, 0)
            self.assertEqual(samples[0].duration_ms, 80)
            self.assertEqual(samples[-1].index, 7)
            self.assertEqual(samples[-1].start_ms, 560)
            self.assertEqual([sample.index for sample in samples], sorted(sample.index for sample in samples))
        finally:
            reader.close()

    def test_renders_resized_rgb_frame(self) -> None:
        reader = AnimationReader(make_animated_gif(), "gif", max_scan_frames=3000)
        try:
            frame = reader.render_frame(0, max_edge=32, background="white")
            self.assertEqual(frame.mode, "RGB")
            self.assertLessEqual(max(frame.size), 32)
        finally:
            reader.close()

    def test_renders_rgba_frame_when_keep_alpha(self) -> None:
        reader = AnimationReader(make_animated_gif(), "gif", max_scan_frames=3000)
        try:
            frame = reader.render_frame(0, max_edge=64, background="keep")
            self.assertEqual(frame.mode, "RGBA")
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
