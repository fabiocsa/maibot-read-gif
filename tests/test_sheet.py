"""帧板合成测试。"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

import unittest

from read_animation.frames import AnimationReader
from read_animation.sheet import build_frame_sheet, build_sheet_banner, fit_frame_edge

from tests.image_fixtures import make_animated_gif


class FitFrameEdgeTests(unittest.TestCase):
    """验证帧板像素预算会按需缩小单帧尺寸。"""

    def test_large_sheet_shrinks_edge_within_budget(self) -> None:
        edge = fit_frame_edge(
            width=854,
            height=640,
            frame_count=8,
            columns=4,
            max_edge=512,
            max_sheet_pixels=400_000,
        )
        self.assertLess(edge, 512)
        self.assertGreaterEqual(edge, 128)

    def test_small_sheet_keeps_configured_edge(self) -> None:
        edge = fit_frame_edge(
            width=64,
            height=48,
            frame_count=8,
            columns=4,
            max_edge=512,
            max_sheet_pixels=1_100_000,
        )
        self.assertEqual(edge, 512)

    def test_edge_never_below_minimum(self) -> None:
        edge = fit_frame_edge(
            width=8000,
            height=6000,
            frame_count=16,
            columns=4,
            max_edge=512,
            max_sheet_pixels=100_000,
        )
        self.assertEqual(edge, 128)


class FrameSheetTests(unittest.TestCase):
    """验证帧板输出格式与尺寸。"""

    def test_white_sheet_is_jpeg(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=8, duration=80), "gif", max_scan_frames=3000)
        try:
            samples = reader.sample(8)
            data, media_type = build_frame_sheet(
                reader,
                samples,
                max_edge=64,
                columns=4,
                background="white",
            )
        finally:
            reader.close()

        self.assertEqual(media_type, "image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))
        with Image.open(BytesIO(data)) as sheet:
            self.assertGreater(sheet.width, 200)
            self.assertGreater(sheet.height, 100)

    def test_keep_alpha_sheet_is_png(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=4), "gif", max_scan_frames=3000)
        try:
            samples = reader.sample(4)
            data, media_type = build_frame_sheet(
                reader,
                samples,
                max_edge=64,
                columns=2,
                background="keep",
            )
        finally:
            reader.close()

        self.assertEqual(media_type, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_small_frames_keep_labels_within_cells(self) -> None:
        reader = AnimationReader(
            make_animated_gif(frame_count=2, duration=80, size=(16, 16)),
            "gif",
            max_scan_frames=3000,
        )
        try:
            samples = reader.sample(2)
            data, _ = build_frame_sheet(
                reader,
                samples,
                max_edge=64,
                columns=2,
                background="white",
            )
        finally:
            reader.close()

        measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        max_label_width = max(
            int(measuring_draw.textlength(f"#{index}  t={index * 80}ms  d=80ms")) for index in range(2)
        )
        with Image.open(BytesIO(data)) as sheet:
            self.assertGreaterEqual(sheet.width, 2 * max_label_width)

    def test_banner_reports_timing(self) -> None:
        reader = AnimationReader(make_animated_gif(frame_count=8, duration=80), "gif", max_scan_frames=3000)
        try:
            banner = build_sheet_banner(reader.meta)
        finally:
            reader.close()

        self.assertIn("ANIMATION FRAME SHEET", banner)
        self.assertIn("8 frames", banner)
        self.assertIn("640 ms", banner)
        self.assertIn("LOOP", banner)


if __name__ == "__main__":
    unittest.main()
