import io
import os
import time
import torch
import random
import tarfile
import numpy as np

from torch.utils.data import Dataset


def _np_load_with_retry(path, *, max_attempts=6, base_sleep_s=0.35):
    """Retry np.load on transient I/O (e.g. Google Drive mount EIO)."""
    path = os.fspath(path)
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return np.load(path)
        except OSError as e:
            last_exc = e
            en = getattr(e, "errno", None)
            transient = en == 5 or en == 11 or (
                en is None and "Input/output error" in str(e)
            )
            if transient and attempt + 1 < max_attempts:
                time.sleep(min(base_sleep_s * (2 ** attempt), 8.0))
                continue
            raise
    raise last_exc


def read_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [f.strip() for f in data]
    return data


def _load_from_tar(tar_path, base_path):
    """Load 4 npy (t1n,t1c,t2w,t2f) from tar at base_path, concat along dim=-1."""
    suffixes = ['.t1n.npy', '.t1c.npy', '.t2w.npy', '.t2f.npy']
    arrays = []
    with tarfile.open(tar_path, 'r:gz') as tar:
        for suf in suffixes:
            member = base_path + suf
            f = tar.extractfile(member)
            if f is None:
                raise FileNotFoundError(f"Member {member} not in {tar_path}")
            arr = np.load(io.BytesIO(f.read()))
            arrays.append(arr)
    return np.concatenate(arrays, axis=-1)


class get_custom_dataset(Dataset):
    def __init__(self, dataset_config, partition="train"):

        img_size = dataset_config.img_size
        patch_size = dataset_config.patch_size[0]
        grid_size = [x // patch_size for x in img_size]
        grid_length = grid_size[0] * grid_size[1] * grid_size[2]
        self.img_size = img_size
        self.grid_length = grid_length
        self.attention_type = dataset_config.attention_type
        self.series_length = dataset_config.series_length

        self.spatial = read_txt(dataset_config.spatial_path)
        self.contrast = read_txt(dataset_config.contrast_path)
        self.semantic = read_txt(dataset_config.semantic_path)

        subset_list = ['TCIA', 'RibFrac', 'TotalSegmentator', 'AbdomenCT-1K', 'ISLES2022', 'VerSe', 'amos22',]

        if dataset_config.add_spatial_data:
            ann_spatial = self.spatial
            if dataset_config.is_subset:
                ann_spatial = [x for x in ann_spatial if x.split('/')[-1].split('_')[0] in subset_list]
        else:
            ann_spatial = []

        ann_others = []
        if dataset_config.add_series_data:
            ann_others += self.contrast
            ann_others += self.semantic
        if dataset_config.is_subset:
            ann_others = ann_others[::2]

        self.ann = ann_spatial + ann_others

        # PrefetchingGCSTrainDataLoader 会在每个 batch 前填入 gs:// -> 本地临时路径
        self._gcs_local_rewrite: dict = {}

        length = len(self.ann)
        if partition == "train":
            self.ann = [self.ann[i] for i in range(length) if i % 500 != 0]
        else:
            self.ann = [self.ann[i] for i in range(length) if i % 500 == 0]

    def __len__(self):
        return len(self.ann)

    def _resolve_load_path(self, path_str: str) -> str:
        p = (path_str or "").strip()
        m = getattr(self, "_gcs_local_rewrite", None)
        if m and p in m:
            return m[p]
        return p

    def __getitem__(self, index):
        ann = self.ann[index]

        # Spatial: patch_random_spatial or patch_random_lidc (128^3 single file)
        if 'patch_random_spatial' in ann or 'patch_random_lidc' in ann:
            input_image = _np_load_with_retry(self._resolve_load_path(ann))
            start, stride = random.randint(0, 63), 8
            z_size = self.img_size[2] // self.series_length
            input_image = torch.tensor(input_image)
            input_image = torch.cat((input_image[..., start: start + z_size],
                                     input_image[..., start + stride: start + stride + z_size],
                                     input_image[..., start + 2 * stride: start + 2 * stride + z_size],
                                     input_image[..., start + 3 * stride: start + 3 * stride + z_size]), dim=-1).flatten()
        # BraTS contrast: path.tar.gz:member_prefix (does not match gs://…)
        elif '.tar.gz:' in ann and ',' not in ann:
            tar_part, base_path = ann.split('.tar.gz:', 1)
            tar_path = tar_part + '.tar.gz'
            if os.path.exists(tar_path):
                input_image = _load_from_tar(tar_path, base_path)
                input_image = torch.tensor(input_image).float()
                input_image = input_image.flatten()
            else:
                raise ValueError(f"Tar not found or invalid: {ann[:120]}...")
        # Semantic / legacy contrast: comma-separated paths (4 files)
        elif ',' in ann:
            ann_split_list = ann.split(',')
            for split_id, ann_split in enumerate(ann_split_list):
                input_image_single = _np_load_with_retry(self._resolve_load_path(ann_split.strip()))
                input_image_single = torch.tensor(input_image_single)
                if split_id == 0:
                    input_image = input_image_single
                else:
                    input_image = torch.cat((input_image, input_image_single), dim=-1)
            input_image = input_image.flatten()
        # Inventory spatial: single .npy per line (local path or gcsfuse mount; bare gs:// unsupported)
        elif ann.strip().endswith('.npy'):
            input_image = _np_load_with_retry(self._resolve_load_path(ann.strip()))
            input_image = torch.tensor(input_image.astype(np.float32)).flatten()
        else:
            raise ValueError(f"Unrecognized sample line: {ann[:120]}...")

        input_ids = torch.tensor([1] + [3] * self.grid_length + [2], dtype=torch.int64)
        attention_mask = torch.ones(self.grid_length + 2, self.grid_length + 2, dtype=torch.bool).tril(diagonal=0)
        if self.attention_type == 'prefix':
            prefix_length = random.randint(0, self.grid_length - 1)
        elif self.attention_type == 'causal':
            prefix_length = 0

        prefix_mask = torch.ones(self.grid_length + 2, dtype=torch.bool)
        prefix_mask[:prefix_length + 1] = 0
        attention_mask[:, :prefix_length + 1] = 1
        attention_mask = attention_mask.flatten()

        return {
            "input_ids": np.array(input_ids),
            "input_image": np.array(input_image),
            "attention_mask": np.array(attention_mask),
            "prefix_mask": np.array(prefix_mask),
            "file_path": ann,  # 文件路径，用于训练时打印
        }