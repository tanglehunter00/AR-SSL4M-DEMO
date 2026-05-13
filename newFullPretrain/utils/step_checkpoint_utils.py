"""Step 级断点：模型/优化器/GradScaler + 当前 epoch 的 batch 索引计划 + 已见 npy 路径。"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from utils.gcs_batch_prefetch import _leaf_dataset, drive_paths_in_annotation, gcs_uris_in_annotation

STEP_CK_FORMAT_VERSION = 3


def collect_npy_paths_from_annotation_rows(file_paths) -> List[str]:
    """从 batch 的 file_path（ann 字符串列表）解析 gs:// 与 Drive .npy 路径。"""
    out: List[str] = []
    if file_paths is None:
        return out
    rows = file_paths if isinstance(file_paths, (list, tuple)) else [file_paths]
    for ann in rows:
        s = ann if isinstance(ann, str) else str(ann)
        out.extend(gcs_uris_in_annotation(s))
        out.extend(drive_paths_in_annotation(s))
    return out


def pending_paths_in_remaining_batches(
    dataset,
    epoch_batches: Sequence[Sequence[int]],
    start_batch_idx: int,
    completed: Set[str],
) -> Tuple[List[str], int]:
    """本 epoch 内剩余 batch 涉及的 npy 路径中，尚未出现在 completed 的路径。"""
    leaf = _leaf_dataset(dataset)
    ann_list = getattr(leaf, "ann", None)
    if ann_list is None:
        return [], 0
    pending: Set[str] = set()
    for bi in range(max(0, int(start_batch_idx)), len(epoch_batches)):
        for idx in epoch_batches[bi]:
            if 0 <= idx < len(ann_list):
                pending.update(
                    collect_npy_paths_from_annotation_rows(ann_list[idx])
                )
    pending -= completed
    ordered = sorted(pending)
    return ordered, len(ordered)


def model_param_shapes_summary(model: torch.nn.Module, max_items: int = 500) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, (name, p) in enumerate(model.named_parameters()):
        if i >= max_items:
            rows.append({"_truncated": True, "max_items": max_items})
            break
        rows.append({"name": name, "shape": list(p.shape), "dtype": str(p.dtype)})
    return rows


def save_step_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler_state: Optional[dict],
    epoch: int,
    next_step_in_epoch: int,
    epoch_batches: List[List[int]],
    completed_npy_paths: Set[str],
    best_val_loss: float,
    train_config_snapshot: Dict[str, Any],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = {
        "format_version": STEP_CK_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state,
        "epoch": int(epoch),
        "next_step_in_epoch": int(next_step_in_epoch),
        "epoch_batches": epoch_batches,
        "completed_npy_paths": sorted(completed_npy_paths),
        "best_val_loss": float(best_val_loss),
        "train_config_snapshot": train_config_snapshot,
    }
    torch.save(blob, path)


def write_step_manifest(
    json_path: str,
    *,
    pt_path: str,
    epoch: int,
    next_step_in_epoch: int,
    epoch_batches: List[List[int]],
    completed_npy_paths: Set[str],
    pending_paths: List[str],
    pending_count: int,
    param_summary: List[dict],
) -> None:
    manifest = {
        "pt_file": pt_path,
        "saved_at": datetime.now().isoformat(),
        "epoch_0based": epoch,
        "epoch_1based_display": epoch + 1,
        "next_step_in_epoch": next_step_in_epoch,
        "batches_total_this_epoch": len(epoch_batches),
        "completed_npy_unique_count": len(completed_npy_paths),
        "completed_npy_sample": sorted(completed_npy_paths)[:80],
        "pending_in_remaining_epoch_count": pending_count,
        "pending_in_remaining_epoch_sample": pending_paths[:80],
        "model_parameters_summary": param_summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def resolve_latest_step_checkpoint(output_dir: str) -> Optional[str]:
    root = Path(output_dir) / "step_checkpoints"
    if not root.is_dir():
        return None
    pts = sorted(root.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pts[0].resolve()) if pts else None
