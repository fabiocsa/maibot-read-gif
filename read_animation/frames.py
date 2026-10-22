"""动图解码、元信息收集与等间隔抽帧。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from PIL import Image

AnimationKind = Literal["gif", "webp", "apng"]

DEFAULT_FRAME_DURATION_MS = 100
"""动图帧缺少 duration 字段时使用的兜底时长。"""


class AnimationReadError(RuntimeError):
    """动图无法被 Pillow 读取或解码。"""


@dataclass(frozen=True)
class AnimationMeta:
    """一段动图的元信息。"""

    kind: AnimationKind
    width: int
    height: int
    frame_count: int
    total_duration_ms: int
    loop: int | None
    fps: float
    scan_truncated: bool

    @property
    def duration_seconds(self) -> float:
        """返回总时长（秒）。"""

        return self.total_duration_ms / 1000.0


@dataclass(frozen=True)
class FrameSample:
    """被抽中的一帧及其时间信息。"""

    index: int
    start_ms: int
    duration_ms: int


class AnimationReader:
    """基于 Pillow 的动图读取器。

    同一实例内部持有 ``PIL.Image.Image``，只应在单个同步分析流程内使用，
    使用完毕后调用 :meth:`close` 释放。
    """

    def __init__(self, data: bytes, kind: AnimationKind, *, max_scan_frames: int) -> None:
        """打开动图并收集元信息。

        Args:
            data: 动图文件字节。
            kind: 嗅探得到的格式。
            max_scan_frames: 最多扫描多少帧的时长信息，避免超长动图拖慢解析。
        """

        self._kind = kind
        self._max_scan_frames = max(1, int(max_scan_frames))
        try:
            self._image = Image.open(BytesIO(data))
            self._frame_count = int(getattr(self._image, "n_frames", 1) or 1)
        except Exception as exc:
            raise AnimationReadError(f"无法打开动图: {exc}") from exc

        self._durations: list[int] = []
        self._prefix_ms: list[int] = [0]
        self._scan_truncated = False
        self._collect_durations()
        self._meta = self._build_meta()

    @property
    def meta(self) -> AnimationMeta:
        """返回当前动图的元信息。"""

        return self._meta

    def _collect_durations(self) -> None:
        """逐帧读取 duration，并累积前缀和。"""

        scan_count = min(self._frame_count, self._max_scan_frames)
        for index in range(scan_count):
            try:
                self._image.seek(index)
            except Exception as exc:
                raise AnimationReadError(f"无法读取动图第 {index} 帧: {exc}") from exc

            duration = self._frame_duration_ms(self._image.info)
            self._durations.append(duration)
            self._prefix_ms.append(self._prefix_ms[-1] + duration)

        self._scan_truncated = scan_count < self._frame_count

    @staticmethod
    def _frame_duration_ms(info: dict[str, object]) -> int:
        """从帧信息中提取时长（毫秒）。"""

        raw_duration = info.get("duration")
        try:
            duration = int(raw_duration) if raw_duration is not None else DEFAULT_FRAME_DURATION_MS
        except (TypeError, ValueError):
            duration = DEFAULT_FRAME_DURATION_MS
        return duration if duration > 0 else DEFAULT_FRAME_DURATION_MS

    def _build_meta(self) -> AnimationMeta:
        """根据已扫描的帧信息构造元数据。"""

        scanned_duration_ms = self._prefix_ms[-1]
        fps = 0.0
        if scanned_duration_ms > 0:
            fps = (len(self._durations) / scanned_duration_ms) * 1000.0

        raw_loop = self._image.info.get("loop")
        try:
            loop = int(raw_loop) if raw_loop is not None else None
        except (TypeError, ValueError):
            loop = None

        return AnimationMeta(
            kind=self._kind,
            width=int(self._image.width),
            height=int(self._image.height),
            frame_count=self._frame_count,
            total_duration_ms=scanned_duration_ms,
            loop=loop,
            fps=fps,
            scan_truncated=self._scan_truncated,
        )

    def sample(self, max_frames: int) -> list[FrameSample]:
        """按时间等间隔抽取至多 ``max_frames`` 帧。

        Args:
            max_frames: 期望的采样帧数。

        Returns:
            List[FrameSample]: 按时间顺序排列的采样结果。
        """

        if self._scan_truncated:
            raise AnimationReadError("动图帧数超过解析上限，无法采样")

        indices = pick_frame_indices(self._frame_count, max_frames)
        return [
            FrameSample(
                index=index,
                start_ms=self._prefix_ms[index],
                duration_ms=self._durations[index],
            )
            for index in indices
        ]

    def render_frame(self, index: int, *, max_edge: int, background: str) -> Image.Image:
        """渲染指定帧并缩放到目标长边。

        Args:
            index: 帧序号（从 0 开始）。
            max_edge: 单帧长边像素上限。
            background: ``white`` 表示拍平到白底，``keep`` 表示保留透明通道。

        Returns:
            PIL.Image.Image: 可直接贴到帧板上的图片。

        Raises:
            AnimationReadError: 帧读取失败时抛出。
        """

        if index < 0 or index >= len(self._durations):
            raise AnimationReadError(f"帧序号超出范围: {index}")

        try:
            self._image.seek(index)
            frame = self._image.convert("RGBA")
        except Exception as exc:
            raise AnimationReadError(f"渲染动图第 {index} 帧失败: {exc}") from exc

        edge = max(16, int(max_edge))
        frame.thumbnail((edge, edge), Image.Resampling.LANCZOS)

        if background == "keep":
            return frame

        white_canvas = Image.new("RGBA", frame.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_canvas, frame).convert("RGB")

    def close(self) -> None:
        """释放底层图片句柄。"""

        try:
            self._image.close()
        except Exception:
            pass


def pick_frame_indices(frame_count: int, requested: int) -> list[int]:
    """在总帧数内按时间等间隔挑出采样帧序号。

    Args:
        frame_count: 动图总帧数。
        requested: 期望采样数。

    Returns:
        List[int]: 去重后、按时间顺序排列的帧序号。
    """

    if frame_count <= 0 or requested <= 0:
        return []

    count = min(int(requested), int(frame_count))
    if count <= 1:
        return [0]
    if count >= frame_count:
        return list(range(frame_count))

    candidates = [round(position * (frame_count - 1) / (count - 1)) for position in range(count)]
    indices: list[int] = []
    for candidate in candidates:
        if not indices or indices[-1] != candidate:
            indices.append(candidate)
    return indices
