"""动图描述的本地 JSON 缓存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import json
import logging
import os
import time

CACHE_VERSION = 1


class DescriptionCache:
    """按 key 缓存视觉描述，带淘汰与原子落盘。"""

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int,
        retention_days: int,
        logger: logging.Logger,
    ) -> None:
        """初始化缓存。

        Args:
            path: 缓存 JSON 文件路径。
            max_entries: 最大条目数，0 表示不按数量淘汰。
            retention_days: 保留天数，0 表示不按时间淘汰。
            logger: 插件日志器。
        """

        self._path = path
        self._max_entries = max(0, int(max_entries))
        self._retention_days = max(0, int(retention_days))
        self._logger = logger
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._dirty = False

    async def load(self) -> None:
        """从磁盘加载缓存；文件不存在或损坏时从空缓存开始。"""

        data = await asyncio.to_thread(self._read_cache_file)
        async with self._lock:
            entries = data.get("entries") if isinstance(data, dict) else None
            if isinstance(entries, dict):
                self._entries = {
                    str(key): dict(value)
                    for key, value in entries.items()
                    if isinstance(value, dict) and isinstance(value.get("description"), str)
                }
            self._prune()
            self._dirty = False

        self._logger.debug("动图描述缓存已加载: %s 条", len(self._entries))

    def _read_cache_file(self) -> dict[str, Any] | None:
        """同步读取缓存文件。"""

        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
        except Exception as exc:
            self._logger.warning("读取动图描述缓存失败，将从空缓存开始: %s", exc)
            return None
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            return None
        return payload

    async def get(self, key: str) -> dict[str, Any] | None:
        """读取缓存条目。

        Args:
            key: 缓存键。

        Returns:
            Optional[Dict[str, Any]]: 命中的条目副本；未命中返回 ``None``。
        """

        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry["last_access"] = time.time()
            self._dirty = True
            return dict(entry)

    async def put(self, key: str, *, header: str, description: str, kind: str) -> None:
        """写入一条缓存。

        Args:
            key: 缓存键。
            header: 注入用元信息头。
            description: 视觉描述。
            kind: 动图格式。
        """

        now = time.time()
        async with self._lock:
            self._entries[key] = {
                "header": header,
                "description": description,
                "kind": kind,
                "created_at": now,
                "last_access": now,
            }
            self._prune()
            self._dirty = True

    def _prune(self) -> None:
        """按时间与数量淘汰缓存条目。"""

        now = time.time()
        if self._retention_days > 0:
            expire_before = now - self._retention_days * 86400
            self._entries = {
                key: entry
                for key, entry in self._entries.items()
                if float(entry.get("created_at", 0.0)) >= expire_before
            }

        if self._max_entries > 0 and len(self._entries) > self._max_entries:
            ranked = sorted(self._entries.items(), key=lambda item: float(item[1].get("last_access", 0.0)))
            drop_count = len(self._entries) - self._max_entries
            self._entries = dict(ranked[drop_count:])

    async def save(self) -> None:
        """把缓存原子写入磁盘（文件内容未变化时跳过）。"""

        if not self._dirty:
            return
        async with self._lock:
            payload = {
                "version": CACHE_VERSION,
                "entries": {key: dict(value) for key, value in self._entries.items()},
            }
            self._dirty = False
        await asyncio.to_thread(self._write_cache_file, payload)

    def _write_cache_file(self, payload: dict[str, Any]) -> None:
        """同步写入缓存文件。"""

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            with temporary_path.open("w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self._path)
        except Exception as exc:
            self._logger.warning("写入动图描述缓存失败: %s", exc)

    async def close(self) -> None:
        """保存并释放缓存。"""

        await self.save()
