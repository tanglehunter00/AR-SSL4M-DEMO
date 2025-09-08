import inspect
from dataclasses import asdict

import torch.distributed as dist
from torch.utils.data import DistributedSampler

from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

from configs import datasets, train_config
from data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from utils.dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()

    update_config(dataset_config, **kwargs)

    return dataset_config


def get_dataloader_kwargs(train_config, dataset, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        if train_config.batching_strategy == "padding":
            if train_config.enable_fsdp:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=mode=="train",
                )
            else:
                kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=mode=="train", shuffle=mode=="train")
            # kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
            kwargs["collate_fn"] = default_data_collator
        elif train_config.batching_strategy == "packing":
            if train_config.enable_fsdp:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

        return kwargs
