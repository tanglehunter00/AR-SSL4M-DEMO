from dataclasses import dataclass


@dataclass
class train_config:
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=24
    batching_strategy: str="padding"
    # 预取模式：先下载若干 batch 到磁盘，训练时异步预取，用后即删，避免 Colab 磁盘占满
    use_prefetch: bool=False
    prefetch_cache_dir: str="/tmp/ar_ssl4m_prefetch_cache"
    prefetch_buffer_batches: int=3  # 预下载 batch 数（step0,1,2），训练 step0 时拉 step3
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
    output_dir: str="PATH/to/save/model"
    freeze_layers: bool=False
    num_freeze_layers: int=1
    save_model: bool=True
    save_optimizer: bool=False
    save_metrics: bool=False
    scheduler:str='CosineLR'
    min_lr: float=0
    pos_type: str='sincos3d'
    norm_pixel_loss: bool=True
