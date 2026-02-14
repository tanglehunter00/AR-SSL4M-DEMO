import io
import torch
import random
import numpy as np

from torch.utils.data import Dataset

try:
    import requests
except ImportError:
    requests = None


def read_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [f.strip() for f in data]
    return data


def _load_npy(source: str, proxy: str = None) -> np.ndarray:
    """
    从本地路径或 URL 加载 .npy 数据。
    URL 模式：通过 requests 拉取到内存，不落盘，用后即释。
    """
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        if requests is None:
            raise ImportError("从 URL 加载数据需要安装 requests: pip install requests PySocks")
        proxies = None
        if proxy:
            proxies = {"http": proxy, "https": proxy}
        r = requests.get(source, proxies=proxies, timeout=30)
        r.raise_for_status()
        return np.load(io.BytesIO(r.content))
    return np.load(source)


def _is_spatial_single_file(ann: str) -> bool:
    """判断是否为单个体积空间数据（本地 patch_random_spatial 或远程 .npy URL）"""
    if "patch_random_spatial" in ann:
        return True
    ann = ann.strip()
    if (ann.startswith("http://") or ann.startswith("https://")) and ".npy" in ann:
        # 远程 URL 单文件，且非逗号分隔的 series
        return "," not in ann
    return False


class get_custom_dataset(Dataset):
    def __init__(self, dataset_config, partition="train"):

        img_size = dataset_config.img_size
        patch_size = dataset_config.patch_size
        if isinstance(patch_size, (list, tuple)):
            grid_size = [img_size[i] // patch_size[i] for i in range(3)]
        else:
            grid_size = [x // patch_size for x in img_size]
        grid_length = grid_size[0] * grid_size[1] * grid_size[2]
        self.img_size = img_size
        self.grid_length = grid_length
        self.attention_type = dataset_config.attention_type
        self.series_length = dataset_config.series_length
        self.fetch_proxy = getattr(dataset_config, "fetch_proxy", None)

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

        length = len(self.ann)
        if partition == "train":
            self.ann = [self.ann[i] for i in range(length) if i % 500 != 0]
        else:
            self.ann = [self.ann[i] for i in range(length) if i % 500 == 0]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if _is_spatial_single_file(ann):
            # 本地 patch_random_spatial 或远程单 .npy URL：按需拉取，4 个 z-slice
            input_image = _load_npy(ann, self.fetch_proxy)
            z_dim = self.img_size[2]
            z_size = z_dim // self.series_length
            stride = z_size
            max_start = max(0, z_dim - 3 * stride - z_size)
            start = random.randint(0, max_start) if max_start > 0 else 0
            input_image = torch.tensor(input_image)
            input_image = torch.cat((input_image[..., start: start + z_size],
                                     input_image[..., start + stride: start + stride + z_size],
                                     input_image[..., start + 2 * stride: start + 2 * stride + z_size],
                                     input_image[..., start + 3 * stride: start + 3 * stride + z_size]), dim=-1).flatten()
        else:
            ann_split_list = ann.split(',')
            for split_id, ann_split in enumerate(ann_split_list):
                input_image_single = _load_npy(ann_split.strip(), self.fetch_proxy)
                input_image_single = torch.tensor(input_image_single)
                if split_id == 0:
                    input_image = input_image_single
                else:
                    input_image = torch.cat((input_image, input_image_single), dim=-1)
            input_image = input_image.flatten()

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
        }