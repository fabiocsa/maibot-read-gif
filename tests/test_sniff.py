"""动图格式嗅探测试。"""

from __future__ import annotations

import unittest

from read_animation.sniff import detect_animation_kind

from tests.image_fixtures import (
    make_animated_gif,
    make_animated_png,
    make_animated_webp,
    make_broken_gif,
    make_static_jpeg,
    make_static_png,
)


class DetectAnimationKindTests(unittest.TestCase):
    """验证 GIF/WebP/APNG 的容器层识别。"""

    def test_detects_gif(self) -> None:
        self.assertEqual(detect_animation_kind(make_animated_gif()), "gif")

    def test_detects_webp(self) -> None:
        self.assertEqual(detect_animation_kind(make_animated_webp()), "webp")

    def test_detects_apng(self) -> None:
        data = make_animated_png()
        if b"acTL" not in data:
            self.skipTest("当前 Pillow 未写出 APNG 控制块")
        self.assertEqual(detect_animation_kind(data), "apng")

    def test_static_png_is_not_animation(self) -> None:
        self.assertIsNone(detect_animation_kind(make_static_png()))

    def test_static_jpeg_is_not_animation(self) -> None:
        self.assertIsNone(detect_animation_kind(make_static_jpeg()))

    def test_broken_gif_header_is_still_detected_as_gif(self) -> None:
        # 嗅探只判断容器类型，内容损坏交由后续解码层处理。
        self.assertEqual(detect_animation_kind(make_broken_gif()), "gif")

    def test_short_bytes_are_ignored(self) -> None:
        self.assertIsNone(detect_animation_kind(b"GIF89a"))
        self.assertIsNone(detect_animation_kind(b""))


if __name__ == "__main__":
    unittest.main()
