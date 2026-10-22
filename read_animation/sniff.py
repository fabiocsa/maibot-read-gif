"""动图容器格式嗅探。

只用文件头与容器块判断类型，不做像素解码；真正的帧数由 ``frames.AnimationReader`` 复核。
"""

from __future__ import annotations

from typing import Literal

AnimationKind = Literal["gif", "webp", "apng"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_RIFF_HEADER_SIZE = 12


def detect_animation_kind(data: bytes) -> AnimationKind | None:
    """判断二进制内容是否属于受支持的动图格式。

    Args:
        data: 原始文件字节。

    Returns:
        Optional[AnimationKind]: ``"gif"`` / ``"webp"`` / ``"apng"``；静态图或未知格式返回 ``None``。
    """

    if len(data) < _RIFF_HEADER_SIZE:
        return None

    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"

    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp" if _has_webp_anim_chunk(data) else None

    if data.startswith(_PNG_SIGNATURE):
        return "apng" if _has_png_actl_chunk(data) else None

    return None


def _has_webp_anim_chunk(data: bytes) -> bool:
    """遍历 RIFF 子块，判断是否存在动画控制块 ``ANIM``。

    Args:
        data: WebP 文件字节。

    Returns:
        bool: 存在 ``ANIM`` 块时返回 ``True``。
    """

    offset = _RIFF_HEADER_SIZE
    total = len(data)
    while offset + 8 <= total:
        fourcc = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if fourcc == b"ANIM":
            return True
        offset += 8 + chunk_size + (chunk_size & 1)
    return False


def _has_png_actl_chunk(data: bytes) -> bool:
    """遍历 PNG 数据块，判断是否存在 APNG 动画控制块 ``acTL``。

    Args:
        data: PNG 文件字节。

    Returns:
        bool: 存在 ``acTL`` 块时返回 ``True``。
    """

    offset = len(_PNG_SIGNATURE)
    total = len(data)
    while offset + 12 <= total:
        chunk_size = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"acTL":
            return True
        offset += 12 + chunk_size
    return False
