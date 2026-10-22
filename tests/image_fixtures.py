"""测试用的动图样本与消息构造工具。"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw

import base64
import time

FRAME_SIZE = (64, 48)


def _make_frame(index: int, size: tuple[int, int] = FRAME_SIZE) -> Image.Image:
    """生成一帧带序号与移动方块的测试画面。"""

    color = ((index * 37) % 256, (index * 67 + 80) % 256, (index * 23 + 160) % 256)
    frame = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((4, 4, 12 + index * 2, 12), fill=(255, 255, 255))
    draw.text((6, 24), f"F{index}", fill=(255, 255, 255))
    return frame


def _save_animation(frames: list[Image.Image], *, fmt: str, duration: int, loop: int) -> bytes:
    """把多帧保存为动图字节。"""

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format=fmt,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    return buffer.getvalue()


def make_animated_gif(
    frame_count: int = 8,
    duration: int = 80,
    loop: int = 0,
    size: tuple[int, int] = FRAME_SIZE,
) -> bytes:
    """生成指定帧数的 GIF 动图。"""

    frames = [_make_frame(index, size) for index in range(frame_count)]
    return _save_animation(frames, fmt="GIF", duration=duration, loop=loop)


def make_animated_webp(frame_count: int = 6, duration: int = 100, loop: int = 0) -> bytes:
    """生成指定帧数的 WebP 动图。"""

    frames = [_make_frame(index) for index in range(frame_count)]
    return _save_animation(frames, fmt="WEBP", duration=duration, loop=loop)


def make_animated_png(frame_count: int = 5, duration: int = 120, loop: int = 0) -> bytes:
    """生成指定帧数的 APNG 动图。"""

    frames = [_make_frame(index) for index in range(frame_count)]
    return _save_animation(frames, fmt="PNG", duration=duration, loop=loop)


def make_static_png() -> bytes:
    """生成单帧静态 PNG。"""

    buffer = BytesIO()
    _make_frame(0).save(buffer, format="PNG")
    return buffer.getvalue()


def make_static_jpeg() -> bytes:
    """生成单帧静态 JPEG。"""

    buffer = BytesIO()
    _make_frame(0).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_single_frame_gif() -> bytes:
    """生成只有一帧的 GIF。"""

    buffer = BytesIO()
    _make_frame(0).save(buffer, format="GIF")
    return buffer.getvalue()


def make_broken_gif() -> bytes:
    """生成文件头合法但内容损坏的 GIF。"""

    return b"GIF89a" + b"\x00" * 32


def make_image_component(data: bytes, *, component_type: str = "image") -> dict[str, Any]:
    """构造插件 Hook 载荷中的图片/表情组件字典。"""

    return {
        "type": component_type,
        "data": "",
        "hash": "test-hash",
        "binary_data_base64": base64.b64encode(data).decode("ascii"),
    }


def make_message(
    components: list[dict[str, Any]],
    *,
    group_id: str | None = "10001",
    is_mentioned: bool = False,
) -> dict[str, Any]:
    """构造插件 Hook 载荷中的消息字典。"""

    return {
        "message_id": "msg-1",
        "timestamp": str(time.time()),
        "platform": "qq",
        "message_info": {
            "user_info": {
                "user_id": "12345",
                "user_nickname": "测试群友",
                "user_cardname": None,
            },
            "group_info": {"group_id": group_id, "group_name": "测试群"} if group_id else None,
            "additional_config": {},
        },
        "raw_message": components,
        "is_mentioned": is_mentioned,
        "is_at": is_mentioned,
        "is_emoji": False,
        "is_picture": True,
        "is_command": False,
        "is_notify": False,
        "session_id": "session-1",
    }
