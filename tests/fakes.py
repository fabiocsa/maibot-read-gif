"""测试替身：假的视觉模型调用。"""

from __future__ import annotations

from typing import Any

import asyncio

DEFAULT_DESCRIPTION = "一只橘猫先趴着，随后跳起来扑向镜头。"

EMPTY_RESPONSE_RESULT: dict[str, Any] = {
    "success": True,
    "response": "",
    "reasoning": "先分析画面，再组织语言……（思考内容占满 max_tokens）",
}


class FakeLlm:
    """记录调用参数并返回预设结果的假视觉模型。"""

    def __init__(
        self,
        *,
        response: str = DEFAULT_DESCRIPTION,
        error: Exception | None = None,
        responses: list[dict[str, Any]] | None = None,
    ) -> None:
        """初始化假模型。

        Args:
            response: 默认返回的描述文本。
            error: 设置后每次调用都抛出该异常。
            responses: 按顺序消费的预设返回结果；耗尽后回退到 ``response``。
        """

        self.response = response
        self.error = error
        self.responses = list(responses) if responses else None
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """模拟 ``ctx.llm.generate``。"""

        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return {"success": True, "response": self.response}
