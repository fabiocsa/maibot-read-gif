"""测试替身：假的视觉模型调用。"""

from __future__ import annotations

from typing import Any

import asyncio

DEFAULT_DESCRIPTION = "一只橘猫先趴着，随后跳起来扑向镜头。"


class FakeLlm:
    """记录调用参数并返回固定描述的假视觉模型。"""

    def __init__(self, *, response: str = DEFAULT_DESCRIPTION, error: Exception | None = None) -> None:
        """初始化假模型。"""

        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """模拟 ``ctx.llm.generate``。"""

        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return {"success": True, "response": self.response}
