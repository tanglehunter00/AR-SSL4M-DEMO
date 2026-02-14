from dataclasses import dataclass
from typing import Optional


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "image_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    spatial_path: str = r"D:\demo\corrected_train_list.txt"
    contrast_path: str = r"D:\demo\corrected_train_list.txt"
    semantic_path: str = r"D:\demo\corrected_train_list.txt"
    # 远程 URL 模式：通过代理拉取数据，按需加载到内存，用后即释，不占磁盘
    fetch_proxy: Optional[str] = None  # 例如 "socks5h://127.0.0.1:1055"（Tailscale SOCKS5）
    img_size = [128, 128, 32]
    patch_size = [16, 16, 4]
    attention_type = 'prefix'
    add_series_data = False
    add_spatial_data = True
    is_subset = False
    series_length = 4
