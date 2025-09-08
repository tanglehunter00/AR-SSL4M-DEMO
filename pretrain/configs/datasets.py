from dataclasses import dataclass


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "image_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    spatial_path: str = "pretrain/demodata_list.txt"
    contrast_path: str = "pretrain/demodata_list.txt"
    semantic_path: str = "pretrain/demodata_list.txt"
    img_size = [128, 128, 128]
    patch_size = [16, 16, 16]
    attention_type = 'prefix'
    add_series_data = False
    add_spatial_data = True
    is_subset = False
    series_length = 4
