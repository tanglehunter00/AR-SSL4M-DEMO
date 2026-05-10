import os
from dataclasses import dataclass, field


def _pretrain_list_path(filename: str) -> str:
    root = os.environ.get(
        "PRETRAIN_LISTS_ROOT",
        "/content/drive/MyDrive/dataset/pretrain_list",
    )
    return os.path.normpath(os.path.join(root, filename))


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "image_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    spatial_path: str = field(default_factory=lambda: _pretrain_list_path("train_spatial.txt"))
    contrast_path: str = field(default_factory=lambda: _pretrain_list_path("train_contrast.txt"))
    semantic_path: str = field(default_factory=lambda: _pretrain_list_path("train_semantic.txt"))
    img_size = [128, 128, 128]
    patch_size = [16, 16, 16]
    attention_type = 'prefix'
    add_series_data = True
    add_spatial_data = True
    is_subset = False
    series_length = 4
    drive_np_cache_enable: bool = True
    drive_np_cache_dir: str = "/content/drive_np_cache"
    drive_np_cache_max_gb: float = 28.0
