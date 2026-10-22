"""测试用缓存路径与重置工具。

Windows 沙箱下工作区目录带有禁止删除子项的 ACL，因此测试不再创建临时目录，
而是复用 ``tests/.tmp`` 下固定的缓存文件（该目录已被 .gitignore 忽略）。
"""

from __future__ import annotations

from pathlib import Path

import json

TEST_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def ensure_test_tmp_root() -> Path:
    """确保测试临时根目录存在。

    Returns:
        Path: 测试临时根目录。
    """

    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_TMP_ROOT


def cache_file(name: str) -> Path:
    """返回测试临时根目录下的缓存文件路径。

    Args:
        name: 文件名。

    Returns:
        Path: 缓存文件路径。
    """

    return ensure_test_tmp_root() / name


def reset_cache_file(path: Path) -> None:
    """把缓存文件重置为空缓存。

    Args:
        path: 缓存文件路径。
    """

    path.write_text(json.dumps({"version": 1, "entries": {}}), encoding="utf-8")
