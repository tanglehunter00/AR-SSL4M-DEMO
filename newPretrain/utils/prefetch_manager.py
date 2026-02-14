"""
预取管理器：训练前预下载若干 batch 到本地，训练时异步预取后续 batch，用完后删除。

流程：
- 训练前：同步下载 step0、step1、step2（3 个 batch）到 cache_dir
- 训练 step0 时：后台拉取 step3
- step0 训练完：删除 step0 文件，训练 step1，同时拉取 step4
- 若 step1 完成但 step4 未就绪：阻塞等待，直到 step4 下载完成
"""

import random
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch

try:
    import requests
except ImportError:
    requests = None


def _download_url_to_file(url: str, dest_path: str, proxy: Optional[str] = None, timeout: int = 60) -> None:
    """从 URL 下载文件到本地路径。"""
    if requests is None:
        raise ImportError("需要安装 requests 和 PySocks: pip install requests PySocks")
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    r = requests.get(url, proxies=proxies, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


def _process_single_file_to_sample(
    file_path: str,
    img_size: List[int],
    grid_length: int,
    attention_type: str,
    series_length: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    """
    从本地 .npy 文件生成单个样本（与 image_dataset 相同逻辑，但源为本地路径）。
    支持 128×128×32 或 128×128×128 等单体积空间数据，自动按 img_size 计算切片范围。
    """
    input_image = np.load(file_path)
    z_dim = img_size[2]
    z_size = z_dim // series_length
    stride = z_size
    max_start = max(0, z_dim - 3 * stride - z_size)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    input_image = torch.tensor(input_image)
    input_image = torch.cat(
        (
            input_image[..., start : start + z_size],
            input_image[..., start + stride : start + stride + z_size],
            input_image[..., start + 2 * stride : start + 2 * stride + z_size],
            input_image[..., start + 3 * stride : start + 3 * stride + z_size],
        ),
        dim=-1,
    ).flatten()

    if attention_type == "prefix":
        prefix_length = rng.randint(0, grid_length - 1)
    else:
        prefix_length = 0

    input_ids = torch.tensor([1] + [3] * grid_length + [2], dtype=torch.int64)
    attention_mask = torch.ones(
        grid_length + 2, grid_length + 2, dtype=torch.bool
    ).tril(diagonal=0)
    prefix_mask = torch.ones(grid_length + 2, dtype=torch.bool)
    prefix_mask[: prefix_length + 1] = 0
    attention_mask[:, : prefix_length + 1] = 1
    attention_mask = attention_mask.flatten()

    return {
        "input_ids": np.array(input_ids),
        "input_image": np.array(input_image),
        "attention_mask": np.array(attention_mask),
        "prefix_mask": np.array(prefix_mask),
    }


def _collate_batch(samples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """将样本列表 collate 成 batch（与 default_data_collator 效果一致）。"""
    from transformers import default_data_collator

    return default_data_collator(samples)


class PrefetchManager:
    """
    预取管理器：维护磁盘缓存，训练时按序消费 batch，用完后删除并异步预取下一批。
    """

    def __init__(
        self,
        ann_list: List[str],
        batch_size: int,
        buffer_batches: int,
        cache_dir: str,
        proxy: Optional[str],
        dataset_config: Any,
        batch_order: Optional[List[int]] = None,
    ):
        """
        Args:
            ann_list: 训练样本列表（URL 或路径），顺序与 batch 划分一致
            batch_size: 每批样本数
            buffer_batches: 预取缓冲 batch 数（典型 3：step0,1,2 预下载，训练 step0 时拉 step3）
            cache_dir: 本地缓存目录
            proxy: 代理地址，如 socks5h://127.0.0.1:1055
            dataset_config: dataset 配置（img_size, attention_type, series_length 等）
            batch_order: 本 epoch 的 batch 顺序，None 表示 0,1,2,...
        """
        self.ann_list = ann_list
        self.batch_size = batch_size
        self.buffer_batches = buffer_batches
        self.cache_dir = Path(cache_dir)
        self.proxy = proxy
        self.dataset_config = dataset_config

        self.num_batches = len(ann_list) // batch_size
        if batch_order is None:
            batch_order = list(range(self.num_batches))
        self.batch_order = batch_order

        self.img_size = dataset_config.img_size
        patch_size = dataset_config.patch_size[0]
        grid_size = [x // patch_size for x in self.img_size]
        self.grid_length = grid_size[0] * grid_size[1] * grid_size[2]
        self.attention_type = dataset_config.attention_type
        self.series_length = dataset_config.series_length

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ready: Dict[int, bool] = {}
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: List[int] = []
        self._shutdown = False

    def _batch_dir(self, batch_idx: int) -> Path:
        return self.cache_dir / f"batch_{batch_idx}"

    def _download_batch(self, batch_idx: int) -> None:
        """将指定 batch 的样本从 URL 下载到 cache_dir/batch_{idx}/"""
        batch_dir = self._batch_dir(batch_idx)
        batch_dir.mkdir(parents=True, exist_ok=True)
        start = batch_idx * self.batch_size
        urls = self.ann_list[start : start + self.batch_size]
        for i, url in enumerate(urls):
            dest = batch_dir / f"{i}.npy"
            if url.strip().startswith(("http://", "https://")):
                _download_url_to_file(url.strip(), str(dest), self.proxy)
            else:
                # 本地路径，直接复制
                shutil.copy(url.strip(), str(dest))
        with self._lock:
            self._ready[batch_idx] = True

    def _prefetch_worker(self) -> None:
        """后台线程：从队列取 batch_idx 并下载。"""
        while not self._shutdown:
            with self._lock:
                if not self._prefetch_queue:
                    batch_idx = None
                else:
                    batch_idx = self._prefetch_queue.pop(0)
            if batch_idx is None:
                threading.Event().wait(0.5)
                continue
            try:
                self._download_batch(batch_idx)
            except Exception as e:
                with self._lock:
                    self._ready[batch_idx] = False
                raise

    def _ensure_prefetched(self, batch_idx: int) -> None:
        """阻塞直到 batch_idx 已下载完成。"""
        while True:
            with self._lock:
                if self._ready.get(batch_idx, False):
                    return
            import time
            time.sleep(0.2)

    def _start_prefetch(self, batch_idx: int) -> None:
        """将 batch_idx 加入预取队列（后台线程会下载）。"""
        if batch_idx >= self.num_batches:
            return
        with self._lock:
            if not self._ready.get(batch_idx, False) and batch_idx not in self._prefetch_queue:
                self._prefetch_queue.append(batch_idx)

    def reset_for_new_epoch(self, batch_order: Optional[List[int]] = None) -> None:
        """新 epoch 开始：清空状态，可选重排 batch 顺序。"""
        with self._lock:
            self._ready.clear()
            self._prefetch_queue.clear()
        if batch_order is not None:
            self.batch_order = batch_order

    def prepare_initial_batches(self) -> None:
        """训练前：同步下载前 buffer_batches 个 batch（按 batch_order）。"""
        to_download = self.batch_order[: self.buffer_batches]
        for idx in to_download:
            if not self._ready.get(idx, False):
                self._download_batch(idx)
        # 仅在首次启动后台预取线程
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._prefetch_thread.start()

    def get_batch(self, order_pos: int) -> Dict[str, torch.Tensor]:
        """
        获取当前顺序位置的 batch。阻塞直到就绪。
        用完后删除该 batch 的磁盘文件，并触发 order_pos+buffer_batches 的预取。

        Args:
            order_pos: 在 batch_order 中的位置，0 表示第一个要训练的 batch
        Returns:
            与 DataLoader 格式一致的 batch dict
        """
        batch_idx = self.batch_order[order_pos]
        self._ensure_prefetched(batch_idx)

        batch_dir = self._batch_dir(batch_idx)
        rng = random.Random(random.randint(0, 2**31 - 1))
        samples = []
        for i in range(self.batch_size):
            fpath = batch_dir / f"{i}.npy"
            s = _process_single_file_to_sample(
                str(fpath),
                self.img_size,
                self.grid_length,
                self.attention_type,
                self.series_length,
                rng,
            )
            samples.append(s)

        batch = _collate_batch(samples)

        # 删除本 batch 文件
        try:
            shutil.rmtree(batch_dir, ignore_errors=True)
        except Exception:
            pass
        with self._lock:
            self._ready[batch_idx] = False

        # 触发下一个需要的 batch 的预取
        next_pos = order_pos + self.buffer_batches
        if next_pos < len(self.batch_order):
            self._start_prefetch(self.batch_order[next_pos])

        return batch

    def shutdown(self) -> None:
        """停止预取线程并清理缓存目录。"""
        self._shutdown = True
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        except Exception:
            pass


def make_prefetch_iterator_factory(prefetch_manager: PrefetchManager, shuffle: bool):
    """
    返回一个每 epoch 调用的工厂函数，用于预取模式的训练循环。
    用法: factory = make_prefetch_iterator_factory(pm, shuffle=True)
          batch_iterator, num_batches = factory(epoch)
    """

    def factory(epoch: int):
        prefetch_manager.reset_for_new_epoch()
        order = list(range(prefetch_manager.num_batches))
        if shuffle:
            random.shuffle(order)
        prefetch_manager.batch_order = order
        prefetch_manager.prepare_initial_batches()

        def _gen():
            for i in range(prefetch_manager.num_batches):
                yield prefetch_manager.get_batch(i)

        return _gen(), prefetch_manager.num_batches

    return factory
