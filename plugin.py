"""动图帧识别插件入口。

拦截入站消息中的 GIF/WebP/APNG，抽取多帧拼成带时间标注的帧板，
调用宿主视觉模型生成描述，再以文本段形式注入消息，供聊天模型理解动图。
"""

from __future__ import annotations

from typing import Any

from maibot_sdk import (
    CONFIG_RELOAD_SCOPE_SELF,
    HookHandler,
    MaiBotPlugin,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .read_animation.config import ReadAnimationConfig
from .read_animation.pipeline import CACHE_FILE_NAME, AnimationProcessor

HOOK_TIMEOUT_MS = 45000


class ReadAnimationPlugin(MaiBotPlugin):
    """动图帧识别插件。"""

    config_model = ReadAnimationConfig

    def __init__(self) -> None:
        """初始化插件状态。"""

        super().__init__()
        self._processor: AnimationProcessor | None = None

    async def on_load(self) -> None:
        """插件加载：创建处理器并加载描述缓存。"""

        cache_path = self.ctx.paths.data_dir / CACHE_FILE_NAME
        self._processor = AnimationProcessor(
            config=self.config,
            llm_generate=self.ctx.llm.generate,
            logger=self.ctx.logger,
            cache_path=cache_path,
        )
        await self._processor.initialize()
        self.ctx.logger.info(
            "动图帧识别插件已加载（视觉任务: %s）",
            self.config.vlm.task_name or "auto",
        )

    async def on_unload(self) -> None:
        """插件卸载：保存缓存并释放处理器。"""

        if self._processor is not None:
            await self._processor.close()
            self._processor = None
        self.ctx.logger.info("动图帧识别插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        """插件配置热更新：重建缓存与并发控制。"""

        del config_data, version
        if scope != CONFIG_RELOAD_SCOPE_SELF or self._processor is None:
            return
        await self._processor.update_config(self.config)
        self.ctx.logger.info("动图帧识别配置已更新")

    @HookHandler(
        "chat.receive.before_process",
        name="animation_frame_sheet",
        description="检测入站消息中的 GIF/WebP/APNG，抽帧生成视觉描述并注入消息文本。",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        timeout_ms=HOOK_TIMEOUT_MS,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """在消息进入宿主处理链之前补充动图描述。"""

        del kwargs
        processor = self._processor
        if processor is None or not isinstance(message, dict):
            return None

        try:
            mutated_message = await processor.process_message(message)
        except Exception:
            self.ctx.logger.warning("处理入站消息中的动图时出现异常", exc_info=True)
            return None

        if mutated_message is None:
            return None
        return {"action": "continue", "modified_kwargs": {"message": mutated_message}}


def create_plugin() -> MaiBotPlugin:
    """插件工厂函数，供 Runner 调用。"""

    return ReadAnimationPlugin()
