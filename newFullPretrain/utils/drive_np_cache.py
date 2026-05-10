"""Google Drive 挂载路径的小文件 LRU 缓存：冷启动 copy 到本地盘，热读走缓存。"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

# Colab 典型挂载前缀（可按需扩展）
_DRIVE_MARKERS = ("/content/drive/MyDrive", "/content/drive/")


def is_drive_mount_path(path: str) -> bool:
    norm = os.path.normpath(os.path.abspath(os.fspath(path))).replace("\\", "/")
    return any(marker in norm for marker in _DRIVE_MARKERS)


class DriveNpCache:
    """线程安全 LRU：miss 时 copyfile 进缓存；超出容量按 LRU 删文件。"""

    def __init__(self, cache_dir: str, max_bytes: int):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max(0, int(max_bytes))
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, tuple[str, int]] = OrderedDict()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(sz for _p, sz in self._lru.values())

    def _evict_until_fit(self) -> None:
        total = sum(sz for _p, sz in self._lru.values())
        while total > self._max_bytes and self._lru:
            _key, (path, sz) = self._lru.popitem(last=False)
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
            total -= sz

    def get_local_path(self, src: str, timing_acc: Optional[Dict[str, Any]]) -> str:
        """
        若是 Drive 挂载路径则返回缓存中的本地路径（必要时复制）；
        否则原样返回 src。
        timing_acc 可选；命中不写，miss 累计 drive_cache_copy_s。
        """
        src = os.path.normpath(os.path.abspath(os.fspath(src)))
        if self._max_bytes <= 0 or not is_drive_mount_path(src):
            return src

        key = hashlib.sha256(src.encode("utf-8")).hexdigest()[:48]
        dst = self._cache_dir / f"{key}.npy"

        with self._lock:
            if key in self._lru:
                path, sz = self._lru.pop(key)
                self._lru[key] = (path, sz)
                return path

            t0 = time.perf_counter()
            tmp = dst.with_suffix(".partial")
            try:
                shutil.copyfile(src, tmp)
                tmp.replace(dst)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            elapsed = time.perf_counter() - t0
            if timing_acc is not None:
                timing_acc["drive_cache_copy_s"] = timing_acc.get("drive_cache_copy_s", 0.0) + float(
                    elapsed
                )

            sz = int(dst.stat().st_size)
            self._lru[key] = (str(dst.resolve()), sz)
            self._evict_until_fit()
            return str(dst.resolve())
