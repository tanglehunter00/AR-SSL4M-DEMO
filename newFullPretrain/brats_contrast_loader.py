"""
BraTS 对比训练专用 DataLoader。
从本地缓存读取数据，训练完成后删除已用 npy，用尽的 tar 删除本地 tar.gz。
"""
import random
from typing import Iterator, Dict, Any, Optional

import numpy as np
import torch

from utils.brats_cache_manager import BraTSLocalCacheManager, _load_sample_from_local


def _build_sample(
    local_dir: str,
    base_path: str,
    ann: str,
    img_size: list,
    grid_length: int,
    attention_type: str,
) -> Dict[str, Any]:
    """构建单个样本的 dict，与 image_dataset 的 output 格式一致"""
    arr = _load_sample_from_local(local_dir, base_path)
    input_image = torch.tensor(arr, dtype=torch.float32).flatten()

    input_ids = torch.tensor([1] + [3] * grid_length + [2], dtype=torch.int64)
    attention_mask = torch.ones(grid_length + 2, grid_length + 2, dtype=torch.bool).tril(diagonal=0)
    if attention_type == 'prefix':
        prefix_length = random.randint(0, grid_length - 1)
    else:
        prefix_length = 0

    prefix_mask = torch.ones(grid_length + 2, dtype=torch.bool)
    prefix_mask[:prefix_length + 1] = 0
    attention_mask[:, :prefix_length + 1] = 1
    attention_mask = attention_mask.flatten()

    return {
        "input_ids": input_ids.numpy(),
        "input_image": input_image.numpy(),
        "attention_mask": attention_mask.numpy(),
        "prefix_mask": prefix_mask.numpy(),
        "file_path": ann,
    }


def _collate_batch(samples: list) -> Dict[str, torch.Tensor]:
    """将 batch 内样本 collate 成模型需要的格式"""
    from transformers import default_data_collator
    return default_data_collator(samples)


class BraTSContrastIterator:
    """
    BraTS 对比训练的 batch 迭代器。
    实现与 DataLoader 相同的接口：__iter__、__next__、__len__。
    """

    def __init__(
        self,
        contrast_path: str,
        dataset_config,
        batch_size: int,
        local_cache_root: str = "/content/brats_cache",
        preload_tars: int = 2,
        verbose: bool = True,
    ):
        self.contrast_path = contrast_path
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.local_cache_root = local_cache_root
        self.preload_tars = preload_tars
        self.verbose = verbose

        patch_size = dataset_config.patch_size[0]
        grid_size = [x // patch_size for x in dataset_config.img_size]
        self.grid_length = grid_size[0] * grid_size[1] * grid_size[2]
        self.attention_type = getattr(dataset_config, 'attention_type', 'prefix')

        self._manager = BraTSLocalCacheManager(
            contrast_path=contrast_path,
            local_cache_root=local_cache_root,
            preload_count=preload_tars,
            verbose=verbose,
        )

        self._total_batches = (self._manager.total_samples + batch_size - 1) // batch_size
        self._consumed_batches = 0
        self._last_consumed_tar: Optional[int] = None

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._consumed_batches = 0
        self._last_consumed_tar = None
        self._manager.reset_for_epoch()
        for i in range(min(self.preload_tars, self._manager.total_tars)):
            self._manager.ensure_tar_local(i)
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._consumed_batches >= self._total_batches:
            self._manager.cleanup_all()
            raise StopIteration

        batch_info = self._manager.get_batch_samples(self.batch_size)
        if not batch_info:
            self._manager.cleanup_all()
            raise StopIteration

        samples = []
        for tar_path, base_path, local_dir in batch_info:
            ann = f"{tar_path}:{base_path}"
            s = _build_sample(
                local_dir,
                base_path,
                ann,
                self.dataset_config.img_size,
                self.grid_length,
                self.attention_type,
            )
            samples.append(s)

        batch = _collate_batch(samples)

        self._manager.mark_batch_used(batch_info)
        self._manager.cleanup_finished_tars_from_batch(batch_info)

        self._consumed_batches += 1
        return batch


def create_brats_contrast_loader(
    contrast_path: str,
    dataset_config,
    batch_size: int,
    local_cache_root: str = "/content/brats_cache",
    preload_tars: int = 2,
) -> BraTSContrastIterator:
    """创建 BraTS 对比训练的迭代器"""
    return BraTSContrastIterator(
        contrast_path=contrast_path,
        dataset_config=dataset_config,
        batch_size=batch_size,
        local_cache_root=local_cache_root,
        preload_tars=preload_tars,
    )
