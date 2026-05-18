from dataclasses import dataclass


@dataclass
class train_config:
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=24
    batching_strategy: str="padding"
    gradient_accumulation_steps: int=1
    gradient_clipping: bool=False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=50
    warmup_epochs:int=0
    num_workers_dataloader: int=6
    lr: float=1e-4
    weight_decay: float=0.01
    gamma: float=0.1
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=False
    val_batch_size: int=1
    dataset = "custom_dataset"
    output_dir: str="/content/drive/MyDrive/dataset/pretrain_result/arssl4m"
    freeze_layers: bool=False
    num_freeze_layers: int=1
    save_model: bool=True
    save_optimizer: bool=False
    save_metrics: bool=False
    use_brats_local_cache: bool=True  # BraTS 对比训练时使用本地缓存+异步拉取，不修改 Drive
    use_gcs_batch_prefetch: bool=True  # 默认开启：清单含 gs:// 时按 batch 拉取到本地；纯 BraTS tar/Hybrid 时请改为 False
    gcs_prefetch_ahead_batches: int = 16
    gcs_prefetch_cache_root: str = "/content/gcs_batch_cache"
    gcs_prefetch_download_workers: int = 16
    drive_np_cache_enable: bool = True
    drive_np_cache_dir: str = "/content/drive_np_cache"
    drive_np_cache_max_gb: float = 28.0
    resume_training: bool = False  # True：从 checkpoint 载入模型权重后再训练（优化器仍重新初始化；epoch 计数仍从 1 开始）
    resume_checkpoint_path: str = ""  # 留空则自动选用 output_dir/checkpoints 下 epoch 最大的 *.pth
    scheduler:str='CosineLR'
    min_lr: float=0
    pos_type: str='sincos3d'
    norm_pixel_loss: bool=True
