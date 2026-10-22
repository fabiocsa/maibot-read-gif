"""把采样帧拼成一张带帧号/时间标注的帧板。"""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

from PIL import Image, ImageDraw

import math

from .frames import AnimationMeta, AnimationReader, FrameSample

SHEET_PADDING = 6
SHEET_LABEL_HEIGHT = 16
SHEET_BANNER_HEIGHT = 26
SHEET_JPEG_QUALITY = 88
TEXT_COLOR = (32, 32, 32)
BANNER_COLOR = (16, 16, 16)
MIN_FRAME_EDGE = 128
"""压缩帧板时允许的最小单帧长边。"""
ESTIMATED_LABEL_WIDTH = 150
"""帧号标签的估算宽度（默认位图字体，用于预算帧板尺寸）。"""


def fit_frame_edge(
    *,
    width: int,
    height: int,
    frame_count: int,
    columns: int,
    max_edge: int,
    max_sheet_pixels: int,
) -> int:
    """按帧板像素预算计算实际使用的单帧长边。

    视觉模型的输入开销主要跟帧板尺寸相关；大图 + 多帧会显著拖慢识别。
    当估算帧板像素超过 ``max_sheet_pixels`` 时按面积比例缩小单帧长边。

    Args:
        width: 原动图宽度。
        height: 原动图高度。
        frame_count: 采样帧数。
        columns: 帧板列数。
        max_edge: 配置的单帧长边上限。
        max_sheet_pixels: 帧板总像素预算。

    Returns:
        int: 实际使用的单帧长边（不小于 ``MIN_FRAME_EDGE``）。
    """

    edge = max(MIN_FRAME_EDGE, int(max_edge))
    budget = max(1, int(max_sheet_pixels))
    source_width = max(1, int(width))
    source_height = max(1, int(height))
    column_count = max(1, min(int(columns), max(1, int(frame_count))))
    row_count = math.ceil(max(1, int(frame_count)) / column_count)

    for _ in range(8):
        scale = min(1.0, edge / max(source_width, source_height))
        tile_width = max(1, int(round(source_width * scale)))
        tile_height = max(1, int(round(source_height * scale)))
        cell_width = max(tile_width, ESTIMATED_LABEL_WIDTH)
        sheet_width = column_count * (cell_width + 2 * SHEET_PADDING)
        sheet_height = SHEET_BANNER_HEIGHT + row_count * (tile_height + SHEET_LABEL_HEIGHT + 2 * SHEET_PADDING)
        sheet_pixels = sheet_width * sheet_height
        if sheet_pixels <= budget:
            break
        shrink = math.sqrt(budget / sheet_pixels)
        next_edge = max(MIN_FRAME_EDGE, int(edge * shrink))
        if next_edge >= edge:
            next_edge = max(MIN_FRAME_EDGE, edge - 1)
        edge = next_edge
        if edge <= MIN_FRAME_EDGE:
            break

    return edge


def build_frame_sheet(
    reader: AnimationReader,
    samples: Sequence[FrameSample],
    *,
    max_edge: int,
    columns: int,
    background: str,
) -> tuple[bytes, str]:
    """渲染并编码帧板。

    Args:
        reader: 动图读取器（仍处于打开状态）。
        samples: 采样帧列表。
        max_edge: 单帧长边像素上限。
        columns: 帧板列数。
        background: ``white`` 或 ``keep``，与透明通道处理一致。

    Returns:
        Tuple[bytes, str]: 帧板字节与 MIME 类型（JPEG 或 PNG）。
    """

    tiles: list[tuple[Image.Image, str]] = []
    for sample in samples:
        frame = reader.render_frame(sample.index, max_edge=max_edge, background=background)
        label = f"#{sample.index}  t={sample.start_ms}ms  d={sample.duration_ms}ms"
        tiles.append((frame, label))

    if not tiles:
        raise ValueError("没有可绘制的采样帧")

    meta = reader.meta
    column_count = max(1, min(int(columns), len(tiles)))
    row_count = math.ceil(len(tiles) / column_count)

    measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_widths = [int(measuring_draw.textlength(label)) for _, label in tiles]
    cell_width = max([frame.width for frame, _ in tiles] + label_widths)
    cell_height = max(frame.height for frame, _ in tiles)

    sheet_width = column_count * (cell_width + 2 * SHEET_PADDING)
    sheet_height = SHEET_BANNER_HEIGHT + row_count * (cell_height + SHEET_LABEL_HEIGHT + 2 * SHEET_PADDING)

    keep_alpha = background == "keep"
    canvas_mode = "RGBA" if keep_alpha else "RGB"
    background_color = (255, 255, 255, 255) if keep_alpha else (255, 255, 255)
    canvas = Image.new(canvas_mode, (sheet_width, sheet_height), background_color)
    draw = ImageDraw.Draw(canvas)

    text_color = TEXT_COLOR + (255,) if keep_alpha else TEXT_COLOR
    banner_color = BANNER_COLOR + (255,) if keep_alpha else BANNER_COLOR
    draw.text((SHEET_PADDING, 6), build_sheet_banner(meta), fill=banner_color)

    for position, (frame, label) in enumerate(tiles):
        column = position % column_count
        row = position // column_count
        cell_left = column * (cell_width + 2 * SHEET_PADDING) + SHEET_PADDING
        cell_top = SHEET_BANNER_HEIGHT + row * (cell_height + SHEET_LABEL_HEIGHT + 2 * SHEET_PADDING) + SHEET_PADDING
        offset_x = cell_left + (cell_width - frame.width) // 2
        offset_y = cell_top + (cell_height - frame.height) // 2
        canvas.paste(frame, (offset_x, offset_y))
        label_x = cell_left + (cell_width - label_widths[position]) // 2
        draw.text((label_x, cell_top + cell_height + 2), label, fill=text_color)

    buffer = BytesIO()
    if keep_alpha:
        canvas.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"

    canvas.save(buffer, format="JPEG", quality=SHEET_JPEG_QUALITY, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def build_sheet_banner(meta: AnimationMeta) -> str:
    """构造帧板顶部的说明文字（纯 ASCII，避免字体依赖）。

    Args:
        meta: 动图元信息。

    Returns:
        str: 帧板顶部横幅文本。
    """

    if meta.loop == 0:
        loop_text = "LOOP"
    elif meta.loop is None:
        loop_text = "PLAY ONCE"
    else:
        loop_text = f"REPEAT {meta.loop}"

    return (
        f"ANIMATION FRAME SHEET | {meta.kind.upper()} {meta.width}x{meta.height} | "
        f"{meta.frame_count} frames | {meta.total_duration_ms} ms | {meta.fps:.1f} fps | {loop_text}"
    )
