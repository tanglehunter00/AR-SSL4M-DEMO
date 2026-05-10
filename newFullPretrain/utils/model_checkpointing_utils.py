import torch
import time

from pathlib import Path
from datetime import datetime
from typing import Optional

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    folder_name = (
        cfg.output_dir
        + "/checkpoints"
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None, epoch=0):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.output_dir
        + "/checkpoints"
        + "/"
        + str(epoch)
    )
    save_dir = Path.cwd() / folder_name
    #print(folder_name, save_dir)
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )

def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.output_dir
        + "/checkpoints"
        + "/"
        + str(epoch)
        )
        save_dir = Path.cwd() / folder_name
        # print(folder_name, save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = str(epoch) + ".pth"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def save_model_checkpoint_base(
        model,
        optimizer,
        rank,
        cfg,
        epoch=1,
        best_val_loss: Optional[float] = None,
):
    """保存训练断点：model + optimizer + next_epoch + best_val_loss（format_version=2）。"""
    cpu_state = model.state_dict()
    opt_state = optimizer.state_dict() if optimizer is not None else None
    eff_best = float("inf") if best_val_loss is None else float(best_val_loss)

    print(f"--> saving resume checkpoint (model+optimizer+next_epoch)...")
    folder_name = (
            cfg.output_dir
            + "/checkpoints"
            + "/"
            + str(epoch)
    )
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = str(epoch) + ".pth"
    save_full_path = str(save_dir) + "/" + save_name

    bundle = {
        "format_version": 2,
        "model": cpu_state,
        "optimizer": opt_state,
        "next_epoch": int(epoch) + 1,
        "best_val_loss": eff_best,
    }
    torch.save(bundle, save_full_path)

    print(f"resume checkpoint saved for completed_epoch={epoch} next_epoch={bundle['next_epoch']} at {save_full_path}\n")


def resolve_resume_checkpoint_path(output_dir: str, explicit_path: str):
    """
    返回可用于 torch.load 的 .pth 路径。
    explicit_path 非空则使用该路径（须存在）；
    否则在 output_dir/checkpoints/<epoch>/<epoch>.pth 中选 epoch 最大者。
    """
    explicit_path = (explicit_path or "").strip()
    if explicit_path:
        p = Path(explicit_path)
        return str(p.resolve()) if p.is_file() else None
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return None
    best_ep = -1
    best_path: Optional[Path] = None
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        try:
            ep = int(sub.name)
        except ValueError:
            continue
        cand = sub / f"{ep}.pth"
        if cand.is_file() and ep > best_ep:
            best_ep = ep
            best_path = cand
    return str(best_path.resolve()) if best_path is not None else None


def parse_training_checkpoint(blob):
    """
    解析 .pth：format_version==2 为断点 bundle；否则视为旧版纯 model.state_dict()。
    返回 (model_state_dict, optimizer_state_dict_or_None, next_epoch, best_val_loss, is_legacy)。
    """
    if isinstance(blob, dict) and blob.get("format_version") == 2 and "model" in blob:
        return (
            blob["model"],
            blob.get("optimizer"),
            int(blob.get("next_epoch", 0)),
            float(blob.get("best_val_loss", float("inf"))),
            False,
        )
    return blob, None, 0, float("inf"), True


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    
    print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

   
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.output_dir
        + "/checkpoints"
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + str(epoch) + ".pth"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """


    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")

def load_sharded_model_single_gpu(model, model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model
