"""
BraTS 对比训练本地缓存管理器。
将 tar.gz 从 Google Drive 下载到 Colab 本地，解压后供训练使用。
训练完成后删除本地文件，不修改 Drive 上的原始数据。
跨 tar 取样本时随机选择，增加训练随机性。
"""
import os
import random
import shutil
import tarfile
import threading
import time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np


def _read_contrast_list(contrast_path: str) -> List[Tuple[str, str]]:
    """读取 train_contrast.txt，返回 [(tar_path, base_path), ...]"""
    if not os.path.exists(contrast_path) or os.path.getsize(contrast_path) == 0:
        return []
    with open(contrast_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and ':' in l and ',' not in l]
    out = []
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2 and parts[0].strip().endswith('.tar.gz'):
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def _group_by_tar(entries: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
    """按 tar_path 分组，返回 [(tar_path, [base1, base2, ...]), ...]"""
    groups = defaultdict(list)
    for tar_path, base in entries:
        groups[tar_path].append(base)
    # 按键排序，保证顺序稳定
    sorted_tars = sorted(groups.keys())
    return [(tp, groups[tp]) for tp in sorted_tars]


def _load_sample_from_local(local_extract_dir: str, base_path: str) -> np.ndarray:
    """从本地解压目录加载一个样本（4 个 npy 拼接）"""
    suffixes = ['.t1n.npy', '.t1c.npy', '.t2w.npy', '.t2f.npy']
    arrays = []
    for suf in suffixes:
        p = os.path.join(local_extract_dir, base_path + suf)
        arr = np.load(p)
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1)


def _delete_sample_npy(local_extract_dir: str, base_path: str) -> None:
    """删除一个样本对应的 4 个 npy 文件"""
    suffixes = ['.t1n.npy', '.t1c.npy', '.t2w.npy', '.t2f.npy']
    for suf in suffixes:
        p = os.path.join(local_extract_dir, base_path + suf)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


class BraTSLocalCacheManager:
    """
    管理 BraTS tar 的本地缓存：下载、解压、预取、清理。
    不修改 Google Drive 上的原始 tar.gz。
    """

    def __init__(
        self,
        contrast_path: str,
        local_cache_root: str = "/content/brats_cache",
        preload_count: int = 2,
        verbose: bool = True,
    ):
        self.contrast_path = contrast_path
        self.local_cache_root = local_cache_root
        self.preload_count = preload_count
        self.verbose = verbose

        os.makedirs(local_cache_root, exist_ok=True)

        entries = _read_contrast_list(contrast_path)
        self.tar_samples = _group_by_tar(entries)  # [(tar_path, [base1, base2, ...]), ...]
        self.total_samples = sum(len(bases) for _, bases in self.tar_samples)
        self.total_tars = len(self.tar_samples)

        # 当前状态
        self._local_tars: Dict[int, str] = {}  # tar_idx -> local_extract_dir
        self._local_tar_paths: Dict[int, str] = {}  # tar_idx -> local tar.gz path (for deletion)
        self._tar_cursor: int = 0  # 当前消费到的 tar 索引
        self._remaining_indices: Dict[int, List[int]] = {}  # tar_idx -> 未消费的 base 索引，用于随机采样
        self._lock = threading.Lock()
        self._async_download_thread: Optional[threading.Thread] = None
        self._async_download_tar_idx: Optional[int] = None

        if self.verbose:
            print(f"[BraTS Cache] 共 {self.total_tars} 个 tar, {self.total_samples} 个样本")

    def reset_for_epoch(self) -> None:
        """新 epoch 开始时重置消费状态，保留已下载的 tar"""
        self._tar_cursor = 0
        self._remaining_indices.clear()

    def ensure_tar_local(self, tar_idx: int) -> str:
        """确保 tar_idx 对应的 tar 已下载并解压到本地，返回解压目录"""
        with self._lock:
            if tar_idx in self._local_tars:
                return self._local_tars[tar_idx]

        tar_path, bases = self.tar_samples[tar_idx]
        tar_name = os.path.basename(tar_path).replace('.tar.gz', '')
        local_tar = os.path.join(self.local_cache_root, f"{tar_idx}_{tar_name}.tar.gz")
        local_extract = os.path.join(self.local_cache_root, f"{tar_idx}_{tar_name}")

        if os.path.isdir(local_extract) and all(
            os.path.exists(os.path.join(local_extract, b + '.t1n.npy')) for b in bases[:1]
        ):
            with self._lock:
                self._local_tars[tar_idx] = local_extract
                self._local_tar_paths[tar_idx] = local_tar
            return local_extract

        if self.verbose:
            print(f"[BraTS Cache] 下载并解压 tar {tar_idx}/{self.total_tars}: {os.path.basename(tar_path)}")

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Drive 上不存在: {tar_path}")

        shutil.copy2(tar_path, local_tar)
        os.makedirs(local_extract, exist_ok=True)
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(local_extract)

        with self._lock:
            self._local_tars[tar_idx] = local_extract
            self._local_tar_paths[tar_idx] = local_tar

        return local_extract

    def _start_async_download(self, tar_idx: int) -> None:
        """在后台线程中下载并解压下一个 tar"""
        if self._async_download_thread and self._async_download_thread.is_alive():
            return
        if tar_idx >= self.total_tars:
            return

        def _worker():
            try:
                self.ensure_tar_local(tar_idx)
            except Exception as e:
                if self.verbose:
                    print(f"[BraTS Cache] 异步下载 tar {tar_idx} 失败: {e}")

        self._async_download_tar_idx = tar_idx
        self._async_download_thread = threading.Thread(target=_worker, daemon=True)
        self._async_download_thread.start()

    def _wait_async_download(self, tar_idx: int) -> None:
        """等待异步下载完成"""
        if self._async_download_thread and self._async_download_tar_idx == tar_idx:
            self._async_download_thread.join(timeout=600)
            self._async_download_thread = None
            self._async_download_tar_idx = None

    def _get_remaining_for_tar(self, tar_idx: int) -> List[int]:
        """获取 tar 的未消费索引列表，首次访问时初始化"""
        if tar_idx not in self._remaining_indices:
            _, bases = self.tar_samples[tar_idx]
            self._remaining_indices[tar_idx] = list(range(len(bases)))
        return self._remaining_indices[tar_idx]

    def get_remaining_local_samples(self) -> int:
        """当前本地已解压的样本中，尚未消费的数量（仅统计已下载的 tar）"""
        count = 0
        for idx in range(self._tar_cursor, self.total_tars):
            if idx not in self._local_tars:
                break
            count += len(self._get_remaining_for_tar(idx))
        return count

    def ensure_enough_for_batch(self, batch_size: int) -> None:
        """
        确保本地有足够样本供下一次 batch 使用。
        若不足，则启动异步下载下一个 tar；必要时阻塞等待。
        """
        remaining = self.get_remaining_local_samples()
        if remaining >= batch_size:
            return

        next_tar_idx = self._tar_cursor
        while next_tar_idx < self.total_tars and next_tar_idx in self._local_tars:
            next_tar_idx += 1
        while next_tar_idx < self.total_tars:
            self._start_async_download(next_tar_idx)
            self._wait_async_download(next_tar_idx)
            remaining = self.get_remaining_local_samples()
            if remaining >= batch_size:
                return
            next_tar_idx += 1

    def get_batch_samples(self, batch_size: int) -> List[Tuple[str, str, str]]:
        """
        获取下一批样本的 (tar_path, base_path, local_extract_dir)。
        跨 tar 取样本时，从各 tar 的剩余样本中随机选择，增加训练随机性。
        """
        self.ensure_enough_for_batch(batch_size)

        batch = []
        needed = batch_size
        while needed > 0 and self._tar_cursor < self.total_tars:
            tar_idx = self._tar_cursor
            tar_path, bases = self.tar_samples[tar_idx]
            local_dir = self.ensure_tar_local(tar_idx)
            remaining = self._get_remaining_for_tar(tar_idx)
            take = min(needed, len(remaining))

            random.shuffle(remaining)
            take_indices = remaining[:take]
            self._remaining_indices[tar_idx] = remaining[take:]

            for i in take_indices:
                batch.append((tar_path, bases[i], local_dir))
            needed -= take

            if len(self._remaining_indices[tar_idx]) == 0:
                self._tar_cursor += 1
                next_tar = self._tar_cursor + self.preload_count - 1
                if next_tar < self.total_tars:
                    self._start_async_download(next_tar)

            if len(batch) >= batch_size:
                break

        return batch[:batch_size]

    def mark_batch_used(self, batch_info: List[Tuple[str, str, str]]) -> None:
        """标记这批样本已使用，删除对应的本地 npy 文件"""
        for _, base_path, local_dir in batch_info:
            _delete_sample_npy(local_dir, base_path)

    def maybe_cleanup_tar(self, tar_idx: int) -> None:
        """若该 tar 下所有样本已消费并删除，则删除本地 tar.gz"""
        if tar_idx >= self.total_tars:
            return
        _, bases = self.tar_samples[tar_idx]
        local_dir = self._local_tars.get(tar_idx)
        if not local_dir or not os.path.isdir(local_dir):
            return
        all_gone = all(
            not os.path.exists(os.path.join(local_dir, b + '.t1n.npy'))
            for b in bases
        )
        if all_gone:
            local_tar = self._local_tar_paths.get(tar_idx)
            if local_tar and os.path.exists(local_tar):
                try:
                    os.remove(local_tar)
                    if self.verbose:
                        print(f"[BraTS Cache] 已删除本地 tar: {os.path.basename(local_tar)}")
                except OSError:
                    pass
            try:
                shutil.rmtree(local_dir, ignore_errors=True)
            except Exception:
                pass
            with self._lock:
                self._local_tars.pop(tar_idx, None)
                self._local_tar_paths.pop(tar_idx, None)

    def cleanup_finished_tars_from_batch(self, batch_info: List[Tuple[str, str, str]]) -> None:
        """根据本批使用的样本，检查并清理已完全消费的 tar"""
        seen_dirs = set()
        for _, _, local_dir in batch_info:
            if local_dir in seen_dirs:
                continue
            seen_dirs.add(local_dir)
            for tar_idx, ld in list(self._local_tars.items()):
                if ld == local_dir:
                    self.maybe_cleanup_tar(tar_idx)
                    break

    def cleanup_all(self) -> None:
        """清理整个本地缓存目录"""
        if os.path.isdir(self.local_cache_root):
            try:
                shutil.rmtree(self.local_cache_root, ignore_errors=True)
                if self.verbose:
                    print(f"[BraTS Cache] 已清理本地缓存: {self.local_cache_root}")
            except Exception as e:
                if self.verbose:
                    print(f"[BraTS Cache] 清理失败: {e}")
