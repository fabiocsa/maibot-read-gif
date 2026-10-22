"""动图识别管线：检测、抽帧、视觉描述与消息注入。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncio
import base64
import hashlib
import logging
import time

from .cache import DescriptionCache
from .config import FramesSection, ReadAnimationConfig
from .frames import AnimationMeta, AnimationReadError, AnimationReader
from .sheet import build_frame_sheet, fit_frame_edge
from .sniff import AnimationKind, detect_animation_kind
from .vlm import (
    PROMPT_VERSION,
    LlmGenerate,
    VlmDescriptionError,
    VlmEmptyResponseError,
    describe_frame_sheet,
)

HANDLED_COMPONENT_TYPES = frozenset({"image", "emoji"})
MAX_FORWARD_DEPTH = 3
MESSAGE_BUDGET_SECONDS = 40.0
CACHE_FILE_NAME = "description_cache.json"
CAVEAT_TEXT = "（基于等间隔采样帧生成，帧间细节可能有遗漏）"
EMPTY_RESPONSE_RETRY_MIN_TOKENS = 2048
EMPTY_RESPONSE_RETRY_MAX_TOKENS = 8192
SKIP_REASON_TEXT = {
    "too_many_frames": "帧数超出解析上限，未做视觉识别",
    "too_long": "动图时长超出上限，未做视觉识别",
}


class NotAnimatedError(AnimationReadError):
    """图片只有一帧，不属于动图。"""


@dataclass(frozen=True)
class AnimationAnalysis:
    """同步分析阶段的产物。"""

    meta: AnimationMeta
    sheet_bytes: bytes
    sheet_media_type: str
    sampled_count: int
    frame_edge: int = 0
    skip_reason: str = ""


def analyze_animation(
    data: bytes,
    kind: AnimationKind,
    frames_config: FramesSection,
    *,
    build_sheet: bool = True,
) -> AnimationAnalysis:
    """解码动图并按需生成帧板（同步函数，应在线程池中调用）。

    Args:
        data: 动图文件字节。
        kind: 嗅探得到的动图格式。
        frames_config: 抽帧与帧板配置。
        build_sheet: 是否生成帧板；``False`` 时只收集元信息。

    Returns:
        AnimationAnalysis: 分析结果；体量超限时 ``skip_reason`` 非空。

    Raises:
        NotAnimatedError: 图片只有一帧时抛出。
        AnimationReadError: 动图解码失败时抛出。
    """

    reader = AnimationReader(data, kind, max_scan_frames=int(frames_config.max_total_frames))
    try:
        meta = reader.meta
        if meta.frame_count <= 1:
            raise NotAnimatedError("图片只有一帧")

        if meta.scan_truncated or meta.frame_count > int(frames_config.max_total_frames):
            return AnimationAnalysis(
                meta=meta,
                sheet_bytes=b"",
                sheet_media_type="",
                sampled_count=0,
                skip_reason="too_many_frames",
            )

        if meta.total_duration_ms > int(frames_config.max_duration_ms):
            return AnimationAnalysis(
                meta=meta,
                sheet_bytes=b"",
                sheet_media_type="",
                sampled_count=0,
                skip_reason="too_long",
            )

        if not build_sheet:
            return AnimationAnalysis(meta=meta, sheet_bytes=b"", sheet_media_type="", sampled_count=0)

        samples = reader.sample(int(frames_config.max_frames))
        if not samples:
            raise NotAnimatedError("未能抽取到有效帧")

        frame_edge = fit_frame_edge(
            width=meta.width,
            height=meta.height,
            frame_count=len(samples),
            columns=int(frames_config.columns),
            max_edge=int(frames_config.max_edge),
            max_sheet_pixels=int(frames_config.max_sheet_pixels),
        )
        sheet_bytes, sheet_media_type = build_frame_sheet(
            reader,
            samples,
            max_edge=frame_edge,
            columns=int(frames_config.columns),
            background=str(frames_config.background),
        )
        return AnimationAnalysis(
            meta=meta,
            sheet_bytes=sheet_bytes,
            sheet_media_type=sheet_media_type,
            sampled_count=len(samples),
            frame_edge=frame_edge,
        )
    finally:
        reader.close()


class AnimationProcessor:
    """把一条入站消息里的动图替换成"原组件 + 描述文本"的处理器。"""

    def __init__(
        self,
        *,
        config: ReadAnimationConfig,
        llm_generate: LlmGenerate,
        logger: logging.Logger,
        cache_path: Path | None,
    ) -> None:
        """初始化处理器。

        Args:
            config: 插件配置。
            llm_generate: 宿主提供的 ``ctx.llm.generate``。
            logger: 插件日志器。
            cache_path: 缓存文件路径；``None`` 表示不启用缓存。
        """

        self._config = config
        self._llm_generate = llm_generate
        self._logger = logger
        self._cache_path = cache_path
        self._cache: DescriptionCache | None = self._build_cache(cache_path, config)
        self._semaphore = asyncio.Semaphore(max(1, int(config.perf.max_concurrent)))

    async def initialize(self) -> None:
        """加载缓存。"""

        if self._cache is not None:
            await self._cache.load()

    async def close(self) -> None:
        """保存缓存并释放资源。"""

        if self._cache is not None:
            await self._cache.close()

    async def update_config(self, config: ReadAnimationConfig) -> None:
        """配置热更新后重建缓存与并发控制。

        Args:
            config: 新的插件配置。
        """

        old_cache = self._cache
        self._config = config
        self._semaphore = asyncio.Semaphore(max(1, int(config.perf.max_concurrent)))
        if old_cache is not None:
            await old_cache.close()
        self._cache = self._build_cache(self._cache_path, config)
        if self._cache is not None:
            await self._cache.load()

    def _build_cache(self, cache_path: Path | None, config: ReadAnimationConfig) -> DescriptionCache | None:
        """按配置构造缓存对象。"""

        if cache_path is None or not config.cache.enabled:
            return None
        return DescriptionCache(
            cache_path,
            max_entries=int(config.cache.max_entries),
            retention_days=int(config.cache.retention_days),
            logger=self._logger,
        )

    async def process_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """处理一条入站消息。

        Args:
            message: 插件 Hook 收到的消息字典。

        Returns:
            Optional[Dict[str, Any]]: 注入描述后的消息副本；无需修改时返回 ``None``。
        """

        cfg = self._config
        if not cfg.plugin.enabled or cfg.trigger.trigger_mode == "off":
            return None
        if not self._should_process_message(message):
            return None

        raw_message = message.get("raw_message")
        if not isinstance(raw_message, list) or not raw_message:
            return None

        deadline = time.monotonic() + MESSAGE_BUDGET_SECONDS
        budget = {"remaining": max(0, int(cfg.trigger.max_images_per_message))}
        new_raw_message = await self._process_component_list(raw_message, budget, deadline, depth=0)
        if new_raw_message is None:
            return None

        mutated = dict(message)
        mutated["raw_message"] = new_raw_message
        return mutated

    def _should_process_message(self, message: Mapping[str, Any]) -> bool:
        """按触发范围判断是否处理该消息。"""

        trigger_mode = self._config.trigger.trigger_mode
        if trigger_mode == "all":
            return True
        if bool(message.get("is_mentioned")) or bool(message.get("is_at")):
            return True

        message_info = message.get("message_info")
        group_info = message_info.get("group_info") if isinstance(message_info, Mapping) else None
        return not group_info

    async def _process_component_list(
        self,
        components: list[Any],
        budget: dict[str, int],
        deadline: float,
        *,
        depth: int,
    ) -> list[Any] | None:
        """递归处理消息组件列表，返回注入后的新列表或 ``None``。"""

        changed = False
        output: list[Any] = []
        for component in components:
            if not isinstance(component, Mapping):
                output.append(component)
                continue

            component_type = str(component.get("type") or "")
            if component_type == "forward" and depth < MAX_FORWARD_DEPTH:
                rewritten_forward = await self._rewrite_forward(component, budget, deadline, depth + 1)
                if rewritten_forward is not None:
                    output.append(rewritten_forward)
                    changed = True
                    continue

            if component_type in HANDLED_COMPONENT_TYPES:
                injected_text = await self._build_injection(component, budget, deadline)
                if injected_text:
                    output.append(component)
                    output.append({"type": "text", "data": injected_text})
                    changed = True
                    continue

            output.append(component)

        return output if changed else None

    async def _rewrite_forward(
        self,
        component: Mapping[str, Any],
        budget: dict[str, int],
        deadline: float,
        depth: int,
    ) -> Mapping[str, Any] | None:
        """递归处理合并转发节点里的动图。"""

        nodes = component.get("data")
        if not isinstance(nodes, list):
            return None

        changed = False
        rewritten_nodes: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                rewritten_nodes.append(node)
                continue
            content = node.get("content")
            if not isinstance(content, list):
                rewritten_nodes.append(node)
                continue

            new_content = await self._process_component_list(content, budget, deadline, depth=depth)
            if new_content is None:
                rewritten_nodes.append(node)
                continue

            rewritten_node = dict(node)
            rewritten_node["content"] = new_content
            rewritten_nodes.append(rewritten_node)
            changed = True

        if not changed:
            return None
        rewritten_component = dict(component)
        rewritten_component["data"] = rewritten_nodes
        return rewritten_component

    async def _build_injection(
        self,
        component: Mapping[str, Any],
        budget: dict[str, int],
        deadline: float,
    ) -> str | None:
        """为单个图片/表情组件生成注入文本。"""

        cfg = self._config
        component_type = str(component.get("type") or "")
        if component_type == "image" and not cfg.trigger.handle_images:
            return None
        if component_type == "emoji" and not cfg.trigger.handle_emojis:
            return None

        raw_base64 = component.get("binary_data_base64")
        if not isinstance(raw_base64, str) or not raw_base64:
            return None
        try:
            data = base64.b64decode(raw_base64, validate=False)
        except Exception:
            return None
        if not data or len(data) > int(cfg.frames.max_input_bytes):
            return None

        kind = detect_animation_kind(data)
        if kind is None:
            return None

        cache_key = self._cache_key(data)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return self._format_text(str(cached["header"]), str(cached["description"]))

        can_describe = budget["remaining"] > 0 and deadline > time.monotonic()
        started_at = time.monotonic()
        try:
            analysis = await asyncio.to_thread(
                analyze_animation,
                data,
                kind,
                cfg.frames,
                build_sheet=can_describe,
            )
        except NotAnimatedError:
            return None
        except AnimationReadError as exc:
            self._logger.debug("跳过无法解析的动图: %s", exc)
            return None
        except Exception:
            self._logger.warning("解析动图时出现异常，已跳过", exc_info=True)
            return None

        analyzed_at = time.monotonic()

        header = self._build_header(analysis.meta)
        if analysis.skip_reason:
            skip_text = SKIP_REASON_TEXT.get(analysis.skip_reason, "未做视觉识别")
            return self._format_text(header, "", note=skip_text)

        if not can_describe:
            reason = "已达每条消息识别上限" if budget["remaining"] <= 0 else "处理超时"
            return self._format_text(header, "", note=reason)

        budget["remaining"] -= 1
        description = await self._describe(analysis, header, deadline)
        finished_at = time.monotonic()
        elapsed_ms = (finished_at - started_at) * 1000.0
        analysis_ms = (analyzed_at - started_at) * 1000.0
        vlm_ms = (finished_at - analyzed_at) * 1000.0
        if description is None:
            self._logger.info(
                "动图识别失败，降级为元信息: %s %.0fms（解析 %.0fms / 视觉 %.0fms）",
                kind,
                elapsed_ms,
                analysis_ms,
                vlm_ms,
            )
            return self._format_text(header, "", note="视觉描述不可用")

        await self._store_cached(cache_key, header=header, description=description, kind=kind)
        self._logger.info(
            "动图识别完成: %s %sx%s %s帧 %.0fms（解析 %.0fms / 视觉 %.0fms，帧板 %s列@%spx）",
            kind,
            analysis.meta.width,
            analysis.meta.height,
            analysis.meta.frame_count,
            elapsed_ms,
            analysis_ms,
            vlm_ms,
            cfg.frames.columns,
            analysis.frame_edge,
        )
        return self._format_text(header, description)

    async def _describe(
        self,
        analysis: AnimationAnalysis,
        header: str,
        deadline: float,
    ) -> str | None:
        """在超时与并发限制内调用视觉模型。

        思考模型（如 deepseek-flash-think）会把思考内容计入 ``max_tokens``；
        当正文为空时自动提高额度重试一次，避免直接降级成"视觉描述不可用"。
        """

        cfg = self._config
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 1.0:
            return None

        timeout_seconds = min(cfg.vlm.timeout_ms / 1000.0, remaining_seconds)
        max_tokens = max(64, int(cfg.vlm.max_tokens))

        try:
            return await self._run_vlm(analysis, header, max_tokens, timeout_seconds)
        except VlmEmptyResponseError as exc:
            retry_tokens = min(
                max(max_tokens * 4, EMPTY_RESPONSE_RETRY_MIN_TOKENS),
                EMPTY_RESPONSE_RETRY_MAX_TOKENS,
            )
            retry_seconds = min(timeout_seconds, deadline - time.monotonic())
            if retry_tokens <= max_tokens or retry_seconds <= 3.0:
                self._logger.warning("动图视觉描述为空且无法重试: %s", exc)
                return None

            self._logger.warning(
                "动图视觉描述为空（%s），将 max_tokens %s → %s 重试一次",
                exc,
                max_tokens,
                retry_tokens,
            )
            try:
                return await self._run_vlm(analysis, header, retry_tokens, retry_seconds)
            except asyncio.TimeoutError:
                self._logger.warning("动图视觉描述重试超时: %.1fs", retry_seconds)
                return None
            except VlmDescriptionError as retry_exc:
                self._logger.warning("动图视觉描述重试失败: %s", retry_exc)
                return None
            except Exception:
                self._logger.warning("动图视觉描述重试异常", exc_info=True)
                return None
        except asyncio.TimeoutError:
            self._logger.warning("动图视觉描述超时: %.1fs", timeout_seconds)
            return None
        except VlmDescriptionError as exc:
            self._logger.warning("动图视觉描述失败: %s", exc)
            return None
        except Exception:
            self._logger.warning("动图视觉描述异常", exc_info=True)
            return None

    async def _run_vlm(
        self,
        analysis: AnimationAnalysis,
        header: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        """执行一次视觉模型调用，返回非空描述或抛出对应异常。"""

        cfg = self._config

        async def call_with_semaphore() -> str:
            async with self._semaphore:
                return await describe_frame_sheet(
                    self._llm_generate,
                    sheet_bytes=analysis.sheet_bytes,
                    sheet_media_type=analysis.sheet_media_type,
                    header=header,
                    sampled_count=analysis.sampled_count,
                    total_frames=analysis.meta.frame_count,
                    task_name=cfg.vlm.task_name.strip(),
                    model_name=cfg.vlm.model_name.strip(),
                    temperature=float(cfg.vlm.temperature),
                    max_tokens=max_tokens,
                    max_chars=int(cfg.vlm.max_chars),
                )

        description = await asyncio.wait_for(call_with_semaphore(), timeout=timeout_seconds)
        normalized = description.strip()
        if not normalized:
            raise VlmEmptyResponseError("视觉模型返回空描述")
        return normalized

    def _cache_key(self, data: bytes) -> str:
        """计算“内容 + 影响描述结果的配置”缓存键。"""

        cfg = self._config
        fingerprint = "|".join(
            [
                PROMPT_VERSION,
                cfg.vlm.task_name.strip(),
                cfg.vlm.model_name.strip(),
                str(cfg.vlm.max_chars),
                str(cfg.frames.max_frames),
                str(cfg.frames.max_edge),
                str(cfg.frames.max_sheet_pixels),
                str(cfg.frames.background),
            ]
        )
        content_digest = hashlib.sha256(data).hexdigest()
        config_digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        return f"{content_digest}:{config_digest}"

    async def _get_cached(self, cache_key: str) -> dict[str, Any] | None:
        """读取描述缓存。"""

        if self._cache is None:
            return None
        entry = await self._cache.get(cache_key)
        if entry is None:
            return None
        header = entry.get("header")
        description = entry.get("description")
        if not isinstance(header, str) or not isinstance(description, str):
            return None
        self._logger.debug("动图描述命中缓存: %s", cache_key[:12])
        return entry

    async def _store_cached(self, cache_key: str, *, header: str, description: str, kind: str) -> None:
        """写入描述缓存。"""

        if self._cache is None:
            return
        await self._cache.put(cache_key, header=header, description=description, kind=kind)

    def _format_text(self, header: str, description: str, *, note: str = "") -> str:
        """构造最终注入到聊天中的文本。"""

        cfg = self._config
        prefix = cfg.inject.prefix.strip()
        if cfg.inject.include_metadata and header:
            leading = f"{prefix} {header}".strip() if prefix else header
        else:
            leading = prefix or "[动图]"

        segments = [leading]
        if description:
            segments.append(f"画面：{description}")
            if cfg.inject.include_caveat:
                segments.append(CAVEAT_TEXT)
        elif note:
            segments.append(f"（{note}）")
        else:
            segments.append("（视觉描述不可用）")
        return "；".join(segment for segment in segments if segment)

    @staticmethod
    def _build_header(meta: AnimationMeta) -> str:
        """构造注入文本里的动图元信息头。"""

        if meta.loop == 0:
            loop_text = "循环"
        elif meta.loop is None:
            loop_text = "播放一次"
        else:
            loop_text = f"重复{meta.loop}次"
        fps_text = f"{meta.fps:.0f}" if meta.fps > 0 else "?"
        return (
            f"{meta.kind.upper()} {meta.width}x{meta.height} · {meta.frame_count}帧 · "
            f"{meta.duration_seconds:.1f}s · 约{fps_text}fps · {loop_text}"
        )
