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

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


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


def _file_lock_context(lock_path: str):
    """跨进程文件锁，避免多 worker 同时复制/解压同一 tar"""
    if _HAS_FCNTL:
        lock_dir = os.path.dirname(lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    else:
        yield


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

        t0 = time.perf_counter()
        shutil.copy2(tar_path, local_tar)
        copy_time = time.perf_counter() - t0
        if self.verbose:
            print(f"  [复制 Drive→本地] {os.path.basename(tar_path)}: {copy_time:.2f}s")

        os.makedirs(local_extract, exist_ok=True)
        t0 = time.perf_counter()
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(local_extract)
        extract_time = time.perf_counter() - t0
        if self.verbose:
            print(f"  [解压] {os.path.basename(tar_path)}: {extract_time:.2f}s")

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

    def get_local_status(self) -> List[Tuple[str, int]]:
        """返回当前本地各 tar 的剩余样本数 [(tar_name, 剩余npy组数/样本数), ...]"""
        status = []
        for idx in sorted(self._local_tars.keys()):
            tar_path, bases = self.tar_samples[idx]
            tar_name = os.path.basename(tar_path)
            remaining = len(self._get_remaining_for_tar(idx))
            if remaining > 0:
                status.append((tar_name, remaining))
        return status

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


class BraTSHybridCacheManager:
    """
    混训模式 BraTS 缓存：本地保持 ≥min_local_samples 个样本，
    每次按需从本地 tar 中随机抽取，用后即删（消费），本 epoch 内不会再出现，直到下个 epoch。
    """

    def __init__(
        self,
        contrast_path: str,
        local_cache_root: str = "/content/brats_cache",
        min_local_samples: int = 100,
        preload_tars: int = 2,
        verbose: bool = True,
        keep_tar_for_epoch_reset: bool = True,
    ):
        self.contrast_path = contrast_path
        self.local_cache_root = local_cache_root
        self.min_local_samples = min_local_samples
        self.preload_tars = preload_tars
        self.verbose = verbose
        self.keep_tar_for_epoch_reset = keep_tar_for_epoch_reset

        os.makedirs(local_cache_root, exist_ok=True)

        entries = _read_contrast_list(contrast_path)
        self.tar_samples = _group_by_tar(entries)
        self.total_tars = len(self.tar_samples)
        self.total_samples = sum(len(bases) for _, bases in self.tar_samples)

        self._local_tars: Dict[int, str] = {}
        self._local_tar_paths: Dict[int, str] = {}
        self._remaining: List[Tuple[int, int, str]] = []  # (tar_idx, base_idx_in_tar, local_dir)
        self._lock = threading.Lock()
        self._loaded_tar_range: int = 0  # 已加载的最大 tar 索引+1

        if self.verbose:
            print(f"[BraTS 混训缓存] 共 {self.total_tars} 个 tar, 本地保持 ≥{min_local_samples} 个, 用后即删")

    def _ensure_min_local(self) -> None:
        """确保本地剩余样本 ≥ min_local_samples，不足则预取下一个 tar"""
        with self._lock:
            n = len(self._remaining)
        if n >= self.min_local_samples:
            return

        next_tar = self._loaded_tar_range
        while next_tar < self.total_tars:
            self._load_tar(next_tar)
            next_tar += 1
            with self._lock:
                if len(self._remaining) >= self.min_local_samples:
                    break

    def _load_tar(self, tar_idx: int) -> None:
        """下载并解压 tar，将样本加入 _remaining。使用文件锁避免多 worker 竞态。"""
        if tar_idx >= self.total_tars:
            return
        with self._lock:
            if tar_idx in self._local_tars:
                return

        tar_path, bases = self.tar_samples[tar_idx]
        tar_name = os.path.basename(tar_path).replace('.tar.gz', '')
        local_tar = os.path.join(self.local_cache_root, f"hybrid_{tar_idx}_{tar_name}.tar.gz")
        local_extract = os.path.join(self.local_cache_root, f"hybrid_{tar_idx}_{tar_name}")
        lock_path = os.path.join(self.local_cache_root, f".lock_hybrid_{tar_idx}")

        if os.path.isdir(local_extract) and all(
            os.path.exists(os.path.join(local_extract, b + '.t1n.npy')) for b in bases[:1]
        ):
            with self._lock:
                if tar_idx not in self._local_tars:
                    self._local_tars[tar_idx] = local_extract
                    self._local_tar_paths[tar_idx] = local_tar
                    for i, b in enumerate(bases):
                        self._remaining.append((tar_idx, i, local_extract))
                    self._loaded_tar_range = max(self._loaded_tar_range, tar_idx + 1)
            return

        with _file_lock_context(lock_path):
            with self._lock:
                if tar_idx in self._local_tars:
                    return
            if os.path.isdir(local_extract) and all(
                os.path.exists(os.path.join(local_extract, b + '.t1n.npy')) for b in bases[:1]
            ):
                with self._lock:
                    if tar_idx not in self._local_tars:
                        self._local_tars[tar_idx] = local_extract
                        self._local_tar_paths[tar_idx] = local_tar
                        for i, b in enumerate(bases):
                            self._remaining.append((tar_idx, i, local_extract))
                        self._loaded_tar_range = max(self._loaded_tar_range, tar_idx + 1)
                return

            if self.verbose:
                print(f"[BraTS 混训缓存] 预加载 tar {tar_idx}/{self.total_tars}: {os.path.basename(tar_path)}")

            need_copy = True
            if os.path.exists(local_tar):
                try:
                    sz = os.path.getsize(local_tar)
                    if sz < 1024:
                        os.remove(local_tar)
                    else:
                        need_copy = False
                except OSError:
                    need_copy = True

            if need_copy:
                if not os.path.exists(tar_path):
                    raise FileNotFoundError(f"Drive 上不存在: {tar_path}")
                t0 = time.perf_counter()
                shutil.copy2(tar_path, local_tar)
                if self.verbose:
                    print(f"  [复制] {os.path.basename(tar_path)}: {time.perf_counter()-t0:.2f}s")

            os.makedirs(local_extract, exist_ok=True)
            t0 = time.perf_counter()
            try:
                with tarfile.open(local_tar, 'r:gz') as tar:
                    tar.extractall(local_extract)
            except tarfile.ReadError as e:
                if "empty" in str(e).lower() or "empty file" in str(e).lower():
                    try:
                        os.remove(local_tar)
                        shutil.rmtree(local_extract, ignore_errors=True)
                    except OSError:
                        pass
                    if not os.path.exists(tar_path):
                        raise FileNotFoundError(f"Drive 上不存在: {tar_path}") from e
                    shutil.copy2(tar_path, local_tar)
                    with tarfile.open(local_tar, 'r:gz') as tar:
                        tar.extractall(local_extract)
                else:
                    raise
            if self.verbose:
                print(f"  [解压] {os.path.basename(tar_path)}: {time.perf_counter()-t0:.2f}s")

            with self._lock:
                if tar_idx not in self._local_tars:
                    self._local_tars[tar_idx] = local_extract
                    self._local_tar_paths[tar_idx] = local_tar
                    for i, b in enumerate(bases):
                        self._remaining.append((tar_idx, i, local_extract))
                    self._loaded_tar_range = max(self._loaded_tar_range, tar_idx + 1)

    def get_batch_samples(self, n: int) -> List[Tuple[str, str, str]]:
        """
        从本地剩余样本中随机抽取 n 个，返回 (tar_path, base_path, local_dir)。
        抽取后从池中移除，用完后需调用 mark_batch_used 删除 npy。
        """
        self._ensure_min_local()
        with self._lock:
            pool = list(self._remaining)
        if len(pool) < n:
            raise RuntimeError(f"[BraTS 混训缓存] 本地剩余 {len(pool)} 个样本，需要 {n} 个")

        chosen = random.sample(pool, n)
        base_by_tar = self.tar_samples

        batch = []
        with self._lock:
            chosen_set = set((t, i) for t, i, _ in chosen)
            self._remaining = [(t, i, d) for t, i, d in self._remaining if (t, i) not in chosen_set]

        for tar_idx, base_idx, local_dir in chosen:
            tar_path, bases = base_by_tar[tar_idx]
            base_path = bases[base_idx]
            batch.append((tar_path, base_path, local_dir))

        return batch

    def mark_batch_used(self, batch_info: List[Tuple[str, str, str]]) -> None:
        """标记已使用，删除对应 npy，本 epoch 内不可再用"""
        for _, base_path, local_dir in batch_info:
            _delete_sample_npy(local_dir, base_path)

    def _maybe_cleanup_exhausted_tar(self, local_dir: str) -> None:
        """若某 tar 的样本已全部消费，删除解压目录（保留 tar.gz 以便下个 epoch 重解压）"""
        for tar_idx, ld in list(self._local_tars.items()):
            if ld != local_dir:
                continue
            _, bases = self.tar_samples[tar_idx]
            all_gone = all(
                not os.path.exists(os.path.join(local_dir, b + '.t1n.npy'))
                for b in bases
            )
            if all_gone:
                try:
                    shutil.rmtree(local_dir, ignore_errors=True)
                except Exception:
                    pass
                with self._lock:
                    self._local_tars.pop(tar_idx, None)
                if not self.keep_tar_for_epoch_reset:
                    pt = self._local_tar_paths.pop(tar_idx, None)
                    if pt and os.path.exists(pt):
                        try:
                            os.remove(pt)
                        except OSError:
                            pass
                break

    def cleanup_after_batch(self, batch_info: List[Tuple[str, str, str]]) -> None:
        """根据本批使用的样本，检查并清理已完全消费的 tar 的解压目录"""
        seen = set()
        for _, _, local_dir in batch_info:
            if local_dir not in seen:
                seen.add(local_dir)
                self._maybe_cleanup_exhausted_tar(local_dir)

    def reset_for_epoch(self) -> None:
        """下个 epoch：清空消费状态，若保留 tar.gz 则重新解压前 preload_tars 个 tar"""
        with self._lock:
            self._remaining.clear()
            self._local_tars.clear()
            self._loaded_tar_range = 0

        for i in range(min(self.preload_tars, self.total_tars)):
            self._load_tar(i)
        if self.verbose:
            with self._lock:
                print(f"[BraTS 混训缓存] Epoch 重置完成, 本地剩余 {len(self._remaining)} 个样本")

    def get_local_status(self) -> List[Tuple[str, int]]:
        """[(tar_name, 剩余样本数), ...]"""
        status = []
        with self._lock:
            rem = list(self._remaining)
            tars = dict(self._local_tars)
        for tar_idx in sorted(tars.keys()):
            tar_path, _ = self.tar_samples[tar_idx]
            local_dir = tars[tar_idx]
            cnt = sum(1 for _, _, d in rem if d == local_dir)
            if cnt > 0:
                status.append((os.path.basename(tar_path), cnt))
        return status
