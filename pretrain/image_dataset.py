import torch
import random
import numpy as np
import os

from torch.utils.data import Dataset

# GCS loading: use gcsfs when path is gs://
_GCSFS = None

def _get_gcsfs():
    global _GCSFS
    if _GCSFS is None:
        try:
            import gcsfs
            project = os.environ.get('GOOGLE_CLOUD_PROJECT', None)
            _GCSFS = gcsfs.GCSFileSystem(project=project)
        except ImportError:
            _GCSFS = False
    return _GCSFS

def load_npy(path):
    """Load .npy from local path or GCS (gs://)."""
    if isinstance(path, str) and path.startswith('gs://'):
        fs = _get_gcsfs()
        if fs is False:
            raise ImportError("gcsfs required for GCS paths. Run: pip install gcsfs")
        with fs.open(path, 'rb') as f:
            return np.load(f)
    return np.load(path)


def read_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [f.strip() for f in data]
    return data


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

        length = len(self.ann)
        if partition == "train":
            self.ann = [self.ann[i] for i in range(length) if i % 500 != 0]
        else:
            self.ann = [self.ann[i] for i in range(length) if i % 500 == 0]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if 'patch_random_spatial' in ann or 'patch_random_lidc' in ann:
            input_image = load_npy(ann)
            start, stride = random.randint(0, 63), 8
            z_size = self.img_size[2] // self.series_length
            input_image = torch.tensor(input_image)
            input_image = torch.cat((input_image[..., start: start + z_size],
                                     input_image[..., start + stride: start + stride + z_size],
                                     input_image[..., start + 2 * stride: start + 2 * stride + z_size],
                                     input_image[..., start + 3 * stride: start + 3 * stride + z_size]), dim=-1).flatten()
        else:
            # BraTS contrast: base path (gs:// without .npy) -> expand to 4 modalities
            ann_split_list = ann.split(',')
            if len(ann_split_list) == 1 and ann.strip().startswith('gs://') and '.npy' not in ann.strip():
                base = ann.strip()
                ann_split_list = [f"{base}.t1n.npy", f"{base}.t1c.npy", f"{base}.t2w.npy", f"{base}.t2f.npy"]
            for split_id, ann_split in enumerate(ann_split_list):
                input_image_single = load_npy(ann_split.strip())
                input_image_single = torch.tensor(input_image_single, dtype=torch.float32)
                if split_id == 0:
                    input_image = input_image_single
                else:
                    input_image = torch.cat((input_image, input_image_single), dim=-1)
            # 原始逻辑：4×(128,128,32) concat -> (128,128,128)。若 shape 不符则 resize 以保证混合训练兼容
            target_shape = tuple(self.img_size)
            if tuple(input_image.shape) != target_shape:
                import torch.nn.functional as F
                x = input_image.unsqueeze(0).unsqueeze(0)
                x = F.interpolate(x, size=target_shape, mode='trilinear', align_corners=False)
                input_image = x.squeeze(0).squeeze(0)
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