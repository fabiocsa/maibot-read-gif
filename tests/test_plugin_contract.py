"""插件入口契约测试：按宿主加载方式加载 plugin.py 并实际调用 Hook。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import asyncio
import importlib.util
import json
import logging
import sys
import unittest

from maibot_sdk import MaiBotPlugin

from tests.fakes import DEFAULT_DESCRIPTION, FakeLlm
from tests.image_fixtures import make_animated_gif, make_image_component, make_message
from tests.temp_utils import ensure_test_tmp_root, reset_cache_file

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_MODULE_NAME = "test_read_animation_plugin"


def load_plugin_module() -> Any:
    """按插件运行时的加载方式（synthetic package）加载 plugin.py。"""

    spec = importlib.util.spec_from_file_location(
        PLUGIN_MODULE_NAME,
        str(PLUGIN_DIR / "plugin.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法创建插件模块 spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PLUGIN_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(PLUGIN_MODULE_NAME, None)
        raise
    return module


class PluginContractTests(unittest.TestCase):
    """插件入口、Manifest 与 Hook 行为。"""

    def setUp(self) -> None:
        """加载插件模块。"""

        self._module = load_plugin_module()
        self.addCleanup(sys.modules.pop, PLUGIN_MODULE_NAME, None)

    def test_manifest_is_valid_json_and_v2(self) -> None:
        manifest = json.loads((PLUGIN_DIR / "_manifest.json").read_text(encoding="utf-8"))
        for key in (
            "manifest_version",
            "id",
            "version",
            "name",
            "description",
            "author",
            "license",
            "urls",
            "host_application",
            "sdk",
            "capabilities",
            "i18n",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["manifest_version"], 2)
        self.assertEqual(manifest["id"], "github.fabiocsa.read-animation")
        self.assertEqual(manifest["license"], "MIT")
        self.assertEqual(manifest["plugin_type"], "media")
        self.assertEqual(manifest["changelog"], "CHANGELOG.md")
        self.assertTrue((PLUGIN_DIR / manifest["changelog"]).is_file())
        self.assertTrue((PLUGIN_DIR / "LICENSE").is_file())
        self.assertIn("llm.generate", manifest["capabilities"])
        for version_key in ("host_application", "sdk"):
            for bound in ("min_version", "max_version"):
                parts = str(manifest[version_key][bound]).split(".")
                self.assertEqual(len(parts), 3)

    def test_plugin_contract(self) -> None:
        instance = self._module.create_plugin()
        self.assertIsInstance(instance, MaiBotPlugin)
        self.assertIsNot(type(instance).on_load, MaiBotPlugin.on_load)
        self.assertIsNot(type(instance).on_unload, MaiBotPlugin.on_unload)
        self.assertIsNot(type(instance).on_config_update, MaiBotPlugin.on_config_update)
        self.assertIsNotNone(type(instance).get_config_model())

        components = instance.get_components()
        hook_components = [component for component in components if component["type"] == "HOOK_HANDLER"]
        self.assertEqual(len(hook_components), 1)
        self.assertEqual(hook_components[0]["metadata"]["hook"], "chat.receive.before_process")

        default_config = instance.build_default_config()
        self.assertEqual(default_config["plugin"]["config_version"], "1.2.0")
        self.assertGreaterEqual(default_config["vlm"]["max_tokens"], 2048)
        self.assertGreaterEqual(default_config["frames"]["max_sheet_pixels"], 100000)
        self.assertIn("trigger", default_config)
        self.assertIn("vlm", default_config)

        schema = instance.build_config_schema(
            plugin_id="github.fabiocsa.read-animation",
            plugin_name="动图帧识别",
            plugin_version="0.1.0",
        )
        self.assertIn("sections", schema)
        self.assertIn("plugin", schema["sections"])
        self.assertIn("enabled", schema["sections"]["plugin"]["fields"])

    def test_hook_injects_description(self) -> None:
        instance = self._module.create_plugin()
        instance.set_plugin_config({})

        plugin_data_dir = ensure_test_tmp_root()
        reset_cache_file(plugin_data_dir / "description_cache.json")

        fake_llm = FakeLlm()
        fake_ctx = SimpleNamespace(
            logger=logging.getLogger("read_animation_contract_test"),
            paths=SimpleNamespace(data_dir=plugin_data_dir, runtime_dir=plugin_data_dir),
            llm=SimpleNamespace(generate=fake_llm),
        )
        instance._set_context(fake_ctx)
        message = make_message([make_image_component(make_animated_gif())])

        async def scenario() -> dict[str, Any] | None:
            await instance.on_load()
            try:
                return await instance.handle_before_process(message=message)
            finally:
                await instance.on_unload()

        result = asyncio.run(scenario())

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "continue")  # type: ignore[index]
        modified_message = result["modified_kwargs"]["message"]  # type: ignore[index]
        injected_texts = [
            component["data"]
            for component in modified_message["raw_message"]
            if component.get("type") == "text"
        ]
        self.assertEqual(len(injected_texts), 1)
        self.assertIn("画面：", injected_texts[0])
        self.assertIn(DEFAULT_DESCRIPTION, injected_texts[0])
        self.assertEqual(len(fake_llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
