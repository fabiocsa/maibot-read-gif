"""动图识别插件的配置模型。

Runner 会依据本模块中的模型生成默认 ``config.toml``，并为 WebUI 生成配置表单。
"""

from __future__ import annotations

from typing import Literal

from maibot_sdk import Field, PluginConfigBase


class PluginSection(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用动图识别；关闭后插件不处理任何消息。",
        json_schema_extra={"label": "启用插件"},
    )
    config_version: str = Field(
        default="1.1.0",
        description="配置版本号；修改配置结构时递增，Runner 会据此迁移配置。",
        json_schema_extra={"label": "配置版本"},
    )


class TriggerSection(PluginConfigBase):
    """触发条件配置。"""

    __ui_label__ = "触发条件"
    __ui_order__ = 10

    handle_images: bool = Field(
        default=True,
        description="是否处理 image 类型的动图。",
        json_schema_extra={"label": "处理图片动图"},
    )
    handle_emojis: bool = Field(
        default=True,
        description="是否处理 emoji/表情包类型的动图。",
        json_schema_extra={"label": "处理表情动图"},
    )
    trigger_mode: Literal["all", "mentioned", "off"] = Field(
        default="all",
        description="触发范围：all=所有消息；mentioned=仅被 @、被回复或私聊；off=关闭。",
        json_schema_extra={"label": "触发范围"},
    )
    max_images_per_message: int = Field(
        default=2,
        ge=0,
        le=8,
        description="每条消息最多对几个动图做视觉识别，其余只注入元信息；0 表示只注入元信息、不做识别。",
        json_schema_extra={"label": "每条消息识别上限"},
    )


class FramesSection(PluginConfigBase):
    """抽帧与帧板配置。"""

    __ui_label__ = "抽帧与帧板"
    __ui_order__ = 20

    max_frames: int = Field(
        default=8,
        ge=1,
        le=16,
        description="每段动图最多抽取并展示多少帧。",
        json_schema_extra={"label": "最大抽帧数"},
    )
    max_edge: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="帧板中单帧的长边像素上限，越大越清晰但视觉模型开销越高。",
        json_schema_extra={"label": "单帧长边"},
    )
    columns: int = Field(
        default=4,
        ge=1,
        le=8,
        description="帧板的列数。",
        json_schema_extra={"label": "帧板列数"},
    )
    background: Literal["white", "keep"] = Field(
        default="white",
        description="透明通道处理方式：white=拍平到白底；keep=保留透明（输出 PNG）。",
        json_schema_extra={"label": "透明背景"},
    )
    max_input_bytes: int = Field(
        default=8388608,
        ge=65536,
        le=67108864,
        description="超过该字节数的图片直接跳过，避免过一次大的 RPC 负载。",
        json_schema_extra={"label": "最大输入字节"},
    )
    max_total_frames: int = Field(
        default=3000,
        ge=1,
        description="解析时长信息时最多扫描的帧数，超过时不做视觉识别。",
        json_schema_extra={"label": "最大解析帧数"},
    )
    max_duration_ms: int = Field(
        default=60000,
        ge=1000,
        le=600000,
        description="动图总时长超过该毫秒数时不做视觉识别，只注入元信息。",
        json_schema_extra={"label": "最大动图时长(ms)"},
    )


class VlmSection(PluginConfigBase):
    """视觉模型调用配置。"""

    __ui_label__ = "视觉模型"
    __ui_order__ = 30

    task_name: str = Field(
        default="vlm",
        description="宿主模型任务名，默认使用主程序的视觉任务（model_task_config.vlm）。",
        json_schema_extra={"label": "模型任务"},
    )
    model_name: str = Field(
        default="",
        description="可选：直接指定具体视觉模型名；留空时使用任务配置内的模型列表。",
        json_schema_extra={"label": "指定模型(可选)"},
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="视觉模型温度。",
        json_schema_extra={"label": "温度"},
    )
    max_tokens: int = Field(
        default=4096,
        ge=64,
        le=8192,
        description=(
            "视觉模型最大输出 token 数。若 vlm 任务是思考模型（如 deepseek-flash-think），"
            "思考内容同样占用该额度；512 这类小值会被思考过程吃满导致正文为空，建议不低于 2048。"
        ),
        json_schema_extra={"label": "最大输出 token"},
    )
    timeout_ms: int = Field(
        default=20000,
        ge=1000,
        le=40000,
        description="单次视觉识别超时毫秒数，超时后降级为仅注入元信息。",
        json_schema_extra={"label": "识别超时(ms)"},
    )
    max_chars: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="注入到聊天内容的描述最大字数。",
        json_schema_extra={"label": "描述最大字数"},
    )


class InjectSection(PluginConfigBase):
    """描述注入格式配置。"""

    __ui_label__ = "描述注入"
    __ui_order__ = 40

    prefix: str = Field(
        default="[动图]",
        description="注入文本的前缀，便于聊天模型区分插件生成内容。",
        json_schema_extra={"label": "注入前缀"},
    )
    include_metadata: bool = Field(
        default=True,
        description="是否在注入文本中附带帧数、时长、帧率、循环等元信息。",
        json_schema_extra={"label": "附带元信息"},
    )
    include_caveat: bool = Field(
        default=True,
        description="是否附带“基于采样帧生成、帧间可能有遗漏”的提示。",
        json_schema_extra={"label": "附带采样提示"},
    )


class CacheSection(PluginConfigBase):
    """描述缓存配置。"""

    __ui_label__ = "缓存"
    __ui_order__ = 50

    enabled: bool = Field(
        default=True,
        description="是否缓存同一张动图的视觉描述，命中后不再调用视觉模型。",
        json_schema_extra={"label": "启用缓存"},
    )
    max_entries: int = Field(
        default=1000,
        ge=0,
        le=100000,
        description="缓存的最大条目数，超出后按最近访问时间淘汰。",
        json_schema_extra={"label": "最大缓存条目"},
    )
    retention_days: int = Field(
        default=30,
        ge=0,
        le=3650,
        description="缓存保留天数，0 表示不按时间清理。",
        json_schema_extra={"label": "缓存保留天数"},
    )


class PerfSection(PluginConfigBase):
    """性能与并发配置。"""

    __ui_label__ = "性能"
    __ui_order__ = 60

    max_concurrent: int = Field(
        default=2,
        ge=1,
        le=8,
        description="同时进行的视觉识别请求上限。",
        json_schema_extra={"label": "识别并发上限"},
    )


class ReadAnimationConfig(PluginConfigBase):
    """动图识别插件的完整配置。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    trigger: TriggerSection = Field(default_factory=TriggerSection)
    frames: FramesSection = Field(default_factory=FramesSection)
    vlm: VlmSection = Field(default_factory=VlmSection)
    inject: InjectSection = Field(default_factory=InjectSection)
    cache: CacheSection = Field(default_factory=CacheSection)
    perf: PerfSection = Field(default_factory=PerfSection)
