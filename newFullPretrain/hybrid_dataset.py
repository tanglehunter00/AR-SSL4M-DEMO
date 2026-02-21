"""
混合训练 Dataset：LIDC + BraTS + DeepLesion 混训。
LIDC、DeepLesion 用原逻辑；BraTS 从本地缓存随机抽取，用后即删，本 epoch 内不可再用。
"""
import os
import random
from typing import Optional

import numpy as np
import torch

import shutil

from image_dataset import get_custom_dataset
from utils.brats_cache_manager import BraTSHybridCacheManager, _load_sample_from_local


def _clear_brats_cache_dir(cache_root: str) -> None:
    """清空 BraTS 本地缓存目录，避免残留导致 FileNotFoundError"""
    if os.path.isdir(cache_root):
        try:
            shutil.rmtree(cache_root, ignore_errors=True)
            os.makedirs(cache_root, exist_ok=True)
        except OSError:
            pass


def _is_brats_entry(ann: str) -> bool:
    """判断 ann 是否为 BraTS 格式 (tar_path:base_path)"""
    if not ann or ':' not in ann or ann.count(',') != 0:
        return False
    parts = ann.split(':', 1)
    return len(parts) == 2 and parts[0].strip().endswith('.tar.gz')


def _build_brats_sample(
    local_dir: str,
    base_path: str,
    ann: str,
    img_size: list,
    grid_length: int,
    attention_type: str,
) -> dict:
    """构建单个 BraTS 样本的 dict，与 image_dataset 格式一致"""
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
        "input_ids": np.array(input_ids),
        "input_image": np.array(input_image),
        "attention_mask": np.array(attention_mask),
        "prefix_mask": np.array(prefix_mask),
        "file_path": ann,
    }


class HybridDataset(torch.utils.data.Dataset):
    """
    混训 Dataset：LIDC、DeepLesion 用原逻辑；
    BraTS 从本地缓存随机抽取，用后即删，本 epoch 内不可再用，直到下个 epoch。
    """

    def __init__(
        self,
        dataset_config,
        partition: str = "train",
        local_cache_root: str = "/content/brats_cache",
        min_brats_local: int = 100,
        preload_tars: int = 2,
        verbose: bool = True,
    ):
        self._base = get_custom_dataset(dataset_config, partition)
        self.dataset_config = dataset_config
        self.partition = partition

        patch_size = dataset_config.patch_size[0]
        grid_size = [x // patch_size for x in dataset_config.img_size]
        self.grid_length = grid_size[0] * grid_size[1] * grid_size[2]
        self.attention_type = getattr(dataset_config, 'attention_type', 'prefix')
        self.img_size = dataset_config.img_size

        contrast_path = getattr(dataset_config, 'contrast_path', '')
        self._brats_cache: Optional[BraTSHybridCacheManager] = None
        if contrast_path and os.path.exists(contrast_path) and os.path.getsize(contrast_path) > 0:
            # 启动时清空本地缓存，避免之前运行残留导致 FileNotFoundError
            _clear_brats_cache_dir(local_cache_root)
            self._brats_cache = BraTSHybridCacheManager(
                contrast_path=contrast_path,
                local_cache_root=local_cache_root,
                min_local_samples=min_brats_local,
                preload_tars=preload_tars,
                verbose=verbose,
                keep_tar_for_epoch_reset=False,
            )
        if verbose and self._brats_cache:
            print(f"[HybridDataset] BraTS 从本地缓存随机抽取，用后即删，本地保持 ≥{min_brats_local} 个")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict:
        ann = self._base.ann[index]
        if self._brats_cache and _is_brats_entry(ann):
            batch_info = self._brats_cache.get_batch_samples(1)
            if not batch_info:
                raise RuntimeError("[HybridDataset] BraTS 本地无可用样本")
            tar_path, base_path, local_dir = batch_info[0]
            ann_used = f"{tar_path}:{base_path}"
            sample = _build_brats_sample(
                local_dir,
                base_path,
                ann_used,
                self.img_size,
                self.grid_length,
                self.attention_type,
            )
            self._brats_cache.mark_batch_used(batch_info)
            self._brats_cache.cleanup_after_batch(batch_info)
            return sample
        return self._base[index]
