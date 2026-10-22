"""动图识别管线测试（使用假的视觉模型调用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import unittest

from read_animation.config import ReadAnimationConfig
from read_animation.pipeline import AnimationProcessor
from read_animation.vlm import VlmDescriptionError

from tests.fakes import EMPTY_RESPONSE_RESULT, FakeLlm
from tests.image_fixtures import (
    make_animated_gif,
    make_broken_gif,
    make_image_component,
    make_message,
    make_single_frame_gif,
    make_static_png,
)
from tests.temp_utils import cache_file, reset_cache_file


class AnimationProcessorTests(unittest.IsolatedAsyncioTestCase):
    """覆盖动图检测、注入、缓存、限流与降级。"""

    def setUp(self) -> None:
        """为每个测试准备独立的空缓存。

        工作区沙箱禁止删除子项，因此复用固定缓存文件并在每个测试开始时重置。
        """

        self._cache_path = cache_file("pipeline-cache.json")
        reset_cache_file(self._cache_path)
        self._logger = logging.getLogger("read_animation_test")

    def _build_processor(
        self,
        llm: FakeLlm,
        *,
        config: ReadAnimationConfig | None = None,
        cache_path: Path | None = None,
    ) -> AnimationProcessor:
        """构造处理器。"""

        return AnimationProcessor(
            config=config or ReadAnimationConfig(),
            llm_generate=llm,
            logger=self._logger,
            cache_path=cache_path if cache_path is not None else self._cache_path,
        )

    @staticmethod
    def _texts(message: dict[str, Any]) -> list[str]:
        """取出消息里所有注入的文本。"""

        return [
            str(component.get("data"))
            for component in message["raw_message"]
            if isinstance(component, dict) and component.get("type") == "text"
        ]

    async def test_injects_description_and_keeps_original_component(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            component = make_image_component(make_animated_gif())
            message = make_message([component])

            result = await processor.process_message(message)

            self.assertIsNotNone(result)
            texts = self._texts(result)  # type: ignore[arg-type]
            self.assertEqual(len(texts), 1)
            self.assertIn("画面：", texts[0])
            self.assertIn("跳起来", texts[0])
            self.assertIn("GIF", texts[0])

            rewritten_components = result["raw_message"]  # type: ignore[index]
            self.assertEqual(rewritten_components[0]["binary_data_base64"], component["binary_data_base64"])
            self.assertEqual(rewritten_components[0]["hash"], component["hash"])
            self.assertEqual(len(llm.calls), 1)

            prompt = llm.calls[0]["prompt"]
            self.assertEqual(prompt[0]["role"], "system")
            user_content = prompt[1]["content"]
            self.assertTrue(any(part.get("type") == "image" for part in user_content))
        finally:
            await processor.close()

    async def test_static_image_is_ignored(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_static_png())])
            self.assertIsNone(await processor.process_message(message))
            self.assertEqual(llm.calls, [])
        finally:
            await processor.close()

    async def test_single_frame_gif_is_ignored(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_single_frame_gif())])
            self.assertIsNone(await processor.process_message(message))
            self.assertEqual(llm.calls, [])
        finally:
            await processor.close()

    async def test_broken_gif_is_ignored(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_broken_gif())])
            self.assertIsNone(await processor.process_message(message))
        finally:
            await processor.close()

    async def test_cache_prevents_second_model_call(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        message = make_message([make_image_component(make_animated_gif())])
        try:
            first = await processor.process_message(message)
            self.assertIsNotNone(first)
            self.assertEqual(len(llm.calls), 1)

            second = await processor.process_message(make_message([make_image_component(make_animated_gif())]))
            self.assertIsNotNone(second)
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(self._texts(first), self._texts(second))  # type: ignore[arg-type]
        finally:
            await processor.close()

        second_processor = self._build_processor(llm)
        await second_processor.initialize()
        try:
            third = await second_processor.process_message(make_message([make_image_component(make_animated_gif())]))
            self.assertIsNotNone(third)
            self.assertEqual(len(llm.calls), 1)
        finally:
            await second_processor.close()

    async def test_empty_response_retries_with_more_tokens(self) -> None:
        """思考模型吃满 max_tokens 时，应提高额度重试并拿到描述。"""

        llm = FakeLlm(responses=[EMPTY_RESPONSE_RESULT])
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif())])
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            text = self._texts(result)[0]  # type: ignore[arg-type]
            self.assertIn("画面：", text)
            self.assertIn("跳起来", text)
            self.assertEqual(len(llm.calls), 2)
            self.assertGreater(llm.calls[1]["max_tokens"], llm.calls[0]["max_tokens"])
        finally:
            await processor.close()

    async def test_empty_response_falls_back_when_retry_also_empty(self) -> None:
        """两次都为空正文时降级为元信息，而不是编造内容。"""

        llm = FakeLlm(responses=[EMPTY_RESPONSE_RESULT, EMPTY_RESPONSE_RESULT])
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif())])
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            text = self._texts(result)[0]  # type: ignore[arg-type]
            self.assertIn("视觉描述不可用", text)
            self.assertEqual(len(llm.calls), 2)
        finally:
            await processor.close()

    async def test_vlm_failure_falls_back_to_metadata(self) -> None:
        llm = FakeLlm(error=VlmDescriptionError("模拟失败"))
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif())])
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            text = self._texts(result)[0]  # type: ignore[arg-type]
            self.assertIn("视觉描述不可用", text)
            self.assertIn("GIF", text)
        finally:
            await processor.close()

    async def test_message_budget_limits_model_calls(self) -> None:
        config = ReadAnimationConfig()
        config.trigger.max_images_per_message = 1
        llm = FakeLlm()
        processor = self._build_processor(llm, config=config)
        await processor.initialize()
        try:
            message = make_message(
                [
                    make_image_component(make_animated_gif()),
                    make_image_component(make_animated_gif(frame_count=6, duration=100)),
                ]
            )
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            texts = self._texts(result)  # type: ignore[arg-type]
            self.assertEqual(len(texts), 2)
            self.assertIn("画面：", texts[0])
            self.assertIn("已达每条消息识别上限", texts[1])
            self.assertEqual(len(llm.calls), 1)
        finally:
            await processor.close()

    async def test_emoji_component_is_handled(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif(), component_type="emoji")])
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            self.assertEqual(len(llm.calls), 1)
        finally:
            await processor.close()

    async def test_mentioned_trigger_mode(self) -> None:
        config = ReadAnimationConfig()
        config.trigger.trigger_mode = "mentioned"
        llm = FakeLlm()
        processor = self._build_processor(llm, config=config)
        await processor.initialize()
        try:
            silent_message = make_message([make_image_component(make_animated_gif())], is_mentioned=False)
            self.assertIsNone(await processor.process_message(silent_message))

            mentioned_message = make_message([make_image_component(make_animated_gif())], is_mentioned=True)
            self.assertIsNotNone(await processor.process_message(mentioned_message))
        finally:
            await processor.close()

    async def test_forward_nodes_are_rewritten(self) -> None:
        llm = FakeLlm()
        processor = self._build_processor(llm)
        await processor.initialize()
        try:
            message = make_message(
                [
                    {
                        "type": "forward",
                        "data": [
                            {
                                "user_id": "12345",
                                "user_nickname": "测试群友",
                                "content": [make_image_component(make_animated_gif())],
                            }
                        ],
                    }
                ]
            )
            result = await processor.process_message(message)
            self.assertIsNotNone(result)
            forwarded_content = result["raw_message"][0]["data"][0]["content"]  # type: ignore[index]
            self.assertEqual(forwarded_content[-1]["type"], "text")
            self.assertIn("画面：", forwarded_content[-1]["data"])
        finally:
            await processor.close()

    async def test_disabled_plugin_does_nothing(self) -> None:
        config = ReadAnimationConfig()
        config.plugin.enabled = False
        llm = FakeLlm()
        processor = self._build_processor(llm, config=config)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif())])
            self.assertIsNone(await processor.process_message(message))
        finally:
            await processor.close()

    async def test_image_handling_can_be_disabled(self) -> None:
        config = ReadAnimationConfig()
        config.trigger.handle_images = False
        llm = FakeLlm()
        processor = self._build_processor(llm, config=config)
        await processor.initialize()
        try:
            message = make_message([make_image_component(make_animated_gif())])
            self.assertIsNone(await processor.process_message(message))
        finally:
            await processor.close()


if __name__ == "__main__":
    unittest.main()
