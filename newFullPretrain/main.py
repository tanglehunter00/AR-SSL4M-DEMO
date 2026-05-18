import os
import pdb
import fire
import random
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers.utils import logging

from newNewModel import ReconModel, DecoderLayer
from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from utils.dataset_utils import get_preprocessed_dataset
from utils.gcs_batch_prefetch import PrefetchingGCSTrainDataLoader
from brats_contrast_loader import create_brats_contrast_loader
from hybrid_dataset import HybridDataset
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

logger = logging.get_logger(__name__)


def contrast_list_uses_tar_gz_format(contrast_path: str, max_lines: int = 800) -> bool:
    """train_contrast.txt 若为 BraTS tar 行（*.tar.gz:base）则返回 True；若为逗号分隔 npy 则 False。"""
    try:
        with open(contrast_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                s = line.strip()
                if not s:
                    continue
                if ".tar.gz:" in s and "," not in s:
                    return True
                if "," in s:
                    return False
    except OSError:
        return False
    return False


class Model_Config():
    def __int__(self):
        pass

def main(**kwargs):
    merged = dict(kwargs)
    env_resume = os.environ.get("PRETRAIN_RESUME_TRAINING", "").strip().lower()
    if env_resume in ("1", "true", "yes") and "resume_training" not in merged:
        merged["resume_training"] = True
    rp_env = os.environ.get("PRETRAIN_RESUME_CHECKPOINT_PATH", "").strip()
    if rp_env and "resume_checkpoint_path" not in merged:
        merged["resume_checkpoint_path"] = rp_env

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **merged)
    dataset_config = generate_dataset_config(train_config, merged)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    config = Model_Config()
    config.hidden_size = 768
    config.intermediate_size = 3072
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    config.num_hidden_layers = 12
    config.img_size = dataset_config.img_size
    config.patch_size = dataset_config.patch_size
    config.norm_pixel_loss = train_config.norm_pixel_loss
    config.pos_type = train_config.pos_type

    resume_next_epoch = 0
    resume_best_val_loss = float("inf")
    resume_optimizer_state = None

    model = ReconModel(config)

    if getattr(train_config, "resume_training", False):
        if train_config.enable_fsdp:
            raise RuntimeError(
                "resume_training 当前仅支持 enable_fsdp=False（与本地 bundle checkpoint 对应）。"
            )
        from utils.model_checkpointing_utils import (
            resolve_resume_checkpoint_path,
            parse_training_checkpoint,
        )

        ck_path = resolve_resume_checkpoint_path(
            train_config.output_dir,
            getattr(train_config, "resume_checkpoint_path", "") or "",
        )
        if not ck_path:
            raise FileNotFoundError(
                "resume_training=True 但未找到权重：请设置 resume_checkpoint_path，或确认存在 "
                f"{train_config.output_dir}/checkpoints/<epoch>/<epoch>.pth"
            )
        print(f"--> resume_training: loading checkpoint\n    {ck_path}")
        try:
            blob = torch.load(ck_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ck_path, map_location="cpu")

        model_sd, opt_sd, next_ep, best_v, is_legacy = parse_training_checkpoint(blob)
        incomp = model.load_state_dict(model_sd, strict=False)
        print(
            f"--> load_state_dict(strict=False) missing_keys={len(incomp.missing_keys)} "
            f"unexpected_keys={len(incomp.unexpected_keys)}"
        )

        if is_legacy:
            print(
                "--> 检测到旧版 checkpoint（仅 model 权重）：将从 epoch 1 重新计数，optimizer / best_val 重置。"
            )
            resume_next_epoch = 0
            resume_optimizer_state = None
            resume_best_val_loss = float("inf")
        else:
            resume_optimizer_state = opt_sd
            resume_next_epoch = next_ep
            resume_best_val_loss = best_v
            print(
                f"--> 断点 bundle：next_epoch={resume_next_epoch} "
                f"(将从第 {resume_next_epoch + 1} 个 epoch 继续跑到 num_epochs={train_config.num_epochs}), "
                f"best_val_loss={resume_best_val_loss}"
            )

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)

        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        #model = model.to(rank)
        #model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

    elif not train_config.enable_fsdp:
        model.to("cuda")

    # dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    # BraTS 对比训练专用：使用本地缓存+异步拉取，不修改 Drive
    use_brats_cache = getattr(train_config, "use_brats_local_cache", False)
    contrast_path = getattr(dataset_config, "contrast_path", "")
    add_spatial = getattr(dataset_config, "add_spatial_data", True)
    add_series = getattr(dataset_config, "add_series_data", False)
    contrast_has_data = (
        contrast_path and os.path.exists(contrast_path) and os.path.getsize(contrast_path) > 0
    )
    if contrast_has_data and use_brats_cache and not contrast_list_uses_tar_gz_format(contrast_path):
        use_brats_cache = False
        if not train_config.enable_fsdp or int(os.environ.get("RANK", "0")) == 0:
            print("--> contrast list is comma-separated npy paths; BraTS tar local cache disabled")

    is_brats_contrast_only = (
        use_brats_cache and add_series and contrast_has_data and not add_spatial
    )
    is_hybrid_with_brats_cache = (
        use_brats_cache and contrast_has_data and (add_spatial or add_series)
    )

    if is_brats_contrast_only:
        if not train_config.enable_fsdp or rank == 0:
            print("--> Using BraTS local cache (download to Colab, async prefetch, no Drive modification)")
        train_dataloader = create_brats_contrast_loader(
            contrast_path=contrast_path,
            dataset_config=dataset_config,
            batch_size=train_config.batch_size_training,
            local_cache_root="/content/brats_cache",
            preload_tars=2,
        )
    elif is_hybrid_with_brats_cache:
        if not train_config.enable_fsdp or rank == 0:
            print("--> 混训 (LIDC+BraTS+DeepLesion)，BraTS 从本地缓存随机抽取，用后即删")
        dataset_train = HybridDataset(
            dataset_config,
            partition="train",
            local_cache_root="/content/brats_cache",
            min_brats_local=100,
            preload_tars=2,
            verbose=not train_config.enable_fsdp or rank == 0,
        )
        train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, "train")
        # BraTS 用后即删与多 worker 不兼容：各 worker 共享磁盘文件但各自维护 _remaining，会导致已删样本被其它 worker 再次加载
        train_dl_kwargs["num_workers"] = 0
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            pin_memory=False,
            **train_dl_kwargs,
        )
    elif getattr(train_config, "use_gcs_batch_prefetch", False):
        if train_config.enable_fsdp:
            raise RuntimeError("use_gcs_batch_prefetch 暂不支持 FSDP / 多卡")
        if is_brats_contrast_only or is_hybrid_with_brats_cache:
            raise RuntimeError(
                "use_gcs_batch_prefetch 与 BraTS tar / HybridDataset 互斥；请设置 use_brats_local_cache=False"
            )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"--> GCS batch prefetch: ahead_batches={train_config.gcs_prefetch_ahead_batches}, "
                f"cache_root={train_config.gcs_prefetch_cache_root}, "
                f"download_workers={train_config.gcs_prefetch_download_workers}, "
                f"drive_workers={train_config.gcs_prefetch_drive_workers}"
            )
        train_dataloader = PrefetchingGCSTrainDataLoader(
            dataset_train,
            batch_size=train_config.batch_size_training,
            prefetch_ahead_batches=train_config.gcs_prefetch_ahead_batches,
            local_cache_root=train_config.gcs_prefetch_cache_root,
            max_download_workers=train_config.gcs_prefetch_download_workers,
            max_drive_copy_workers=train_config.gcs_prefetch_drive_workers,
        )
    else:
        train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, "train")
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=False,
            **train_dl_kwargs,
        )

    eval_dataloader = None
    if train_config.run_validation:

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print("--> 已从 checkpoint 恢复 optimizer 状态")
        except Exception as ex:
            print(f"--> 警告：optimizer 未能加载，将使用全新优化器: {ex}")

    if resume_next_epoch >= train_config.num_epochs:
        raise ValueError(
            f"checkpoint 要求从 next_epoch={resume_next_epoch} 继续，但 num_epochs={train_config.num_epochs}；请增大 num_epochs。"
        )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        train_config.gradient_accumulation_steps,
        train_config,
        dataset_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        start_epoch=resume_next_epoch,
        resume_best_val_loss=resume_best_val_loss,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)