"""
Colab 场景：GCS 单小文件直连较慢、本地盘有限。
按与 LengthBasedBatchSampler 相同的打乱与切块方式预先规划 batch，
并行下载当前 batch 之后若干 batch 所需的 gs:// .npy，训练当前 batch 时在后台拉取后续 batch；
每个 batch 用完后仅删除本地缓存文件（不触碰 GCS / Drive）。
"""
from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from transformers import default_data_collator

from data.sampler import LengthBasedBatchSampler


def _leaf_dataset(ds):
    """HybridDataset 等指标挂在 _base 上。"""
    return getattr(ds, "_base", ds)


def _apply_gcs_rewrite(dataset, uri_map: Dict[str, str]) -> None:
    m = dict(uri_map)
    dataset._gcs_local_rewrite = m
    base = getattr(dataset, "_base", None)
    if base is not None:
        base._gcs_local_rewrite = m


def _clear_gcs_rewrite(dataset) -> None:
    dataset._gcs_local_rewrite = {}
    base = getattr(dataset, "_base", None)
    if base is not None:
        base._gcs_local_rewrite = {}


def gcs_uris_in_annotation(ann: str) -> List[str]:
    """从一条样本标注行中收集需要下载的 gs:// 路径。"""
    ann = (ann or "").strip()
    if not ann:
        return []
    out: List[str] = []
    if "," in ann:
        for p in ann.split(","):
            p = p.strip()
            if p.startswith("gs://"):
                out.append(p)
        return out
    if ann.startswith("gs://"):
        out.append(ann)
    return out


def collect_gs_uris_for_indices(dataset, indices: List[int]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for idx in indices:
        for u in gcs_uris_in_annotation(dataset.ann[idx]):
            if u not in seen:
                seen.add(u)
                ordered.append(u)
    return ordered


def _download_gs_uri_to_file(uri: str, dest: Path) -> None:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise ImportError(
            "use_gcs_batch_prefetch 需要安装 google-cloud-storage：pip install google-cloud-storage"
        ) from e

    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri[:80]}")
    rest = uri[5:]
    bucket_name, _, blob_path = rest.partition("/")
    if not bucket_name or not blob_path:
        raise ValueError(f"Bad gs:// URI: {uri[:80]}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(tmp))
    tmp.replace(dest)


def _download_batch_job(
    cache_root: Path,
    epoch_id: int,
    batch_idx: int,
    dataset,
    indices: List[int],
) -> Dict[str, str]:
    """在线程中执行：下载该 batch 所需的全部 gs 对象，返回 gs_uri -> 本地路径。"""
    uris = collect_gs_uris_for_indices(dataset, indices)
    batch_dir = cache_root / f"ep{epoch_id}_b{batch_idx}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    uri_map: Dict[str, str] = {}
    for uri in uris:
        h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:40]
        dst = batch_dir / f"{h}.npy"
        _download_gs_uri_to_file(uri, dst)
        uri_map[uri] = str(dst.resolve())
    return uri_map


class PrefetchingGCSTrainDataLoader:
    """
    行为对齐训练分支：LengthBasedBatchSampler(drop_last=True, shuffle=True)。
    每个 epoch 调用 __iter__ 时会重新随机打乱切 batch。
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        prefetch_ahead_batches: int = 5,
        local_cache_root: str = "/content/gcs_batch_cache",
        max_download_workers: int = 8,
    ):
        self._dataset = dataset
        self.batch_size = batch_size
        self.prefetch_ahead_batches = max(1, int(prefetch_ahead_batches))
        self.cache_root = Path(local_cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_download_workers = max(1, int(max_download_workers))

        self._executor = ThreadPoolExecutor(max_workers=self.max_download_workers)
        self._epoch_id = 0
        self._lock = threading.Lock()
        self.last_batch_io_stats: Dict[str, float] = {}

        if not hasattr(self._dataset, "_gcs_local_rewrite"):
            self._dataset._gcs_local_rewrite = {}

    @property
    def dataset(self):
        return self._dataset

    def __len__(self) -> int:
        n = len(self._dataset)
        return n // self.batch_size

    def _schedule(
        self,
        futures: Dict[int, Future],
        batch_idx: int,
        batches: List[List[int]],
        epoch_tag: int,
    ) -> None:
        if batch_idx >= len(batches) or batch_idx in futures:
            return
        indices = batches[batch_idx]
        fut = self._executor.submit(
            _download_batch_job,
            self.cache_root,
            epoch_tag,
            batch_idx,
            self._dataset,
            indices,
        )
        futures[batch_idx] = fut

    @staticmethod
    def _evict_uri_map(uri_map: Dict[str, str]) -> None:
        if not uri_map:
            return
        batch_dir: Optional[Path] = None
        for _uri, local_p in uri_map.items():
            p = Path(local_p)
            if batch_dir is None:
                batch_dir = p.parent
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        if batch_dir is not None:
            try:
                batch_dir.rmdir()
            except OSError:
                pass

    def __iter__(self):
        with self._lock:
            self._epoch_id += 1
        epoch_id = self._epoch_id

        sampler = LengthBasedBatchSampler(
            self._dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        batches: List[List[int]] = list(iter(sampler))
        n_batches = len(batches)
        futures: Dict[int, Future] = {}

        horizon = self.prefetch_ahead_batches
        for k in range(min(horizon, n_batches)):
            self._schedule(futures, k, batches, epoch_id)

        try:
            for bi, batch_indices in enumerate(batches):
                self._schedule(futures, bi + horizon, batches, epoch_id)

                fut = futures.pop(bi)
                t0 = time.perf_counter()
                uri_map = fut.result()
                gcs_download_s = time.perf_counter() - t0

                leaf = _leaf_dataset(self._dataset)
                if hasattr(leaf, "reset_batch_io_timing"):
                    leaf.reset_batch_io_timing()
                leaf._gcs_staged_root = str(self.cache_root.resolve())

                _apply_gcs_rewrite(self._dataset, uri_map)
                try:
                    t_build = time.perf_counter()
                    samples = [self._dataset[idx] for idx in batch_indices]
                    dataset_wall_s = time.perf_counter() - t_build
                    batch = default_data_collator(samples)
                    acc = getattr(leaf, "_batch_timing_acc", None) or {}
                    self.last_batch_io_stats = {
                        "gcs_download_s": float(gcs_download_s),
                        "dataset_wall_s": float(dataset_wall_s),
                        "drive_cache_copy_s": float(acc.get("drive_cache_copy_s", 0.0)),
                        "gcs_staged_np_load_s": float(acc.get("gcs_staged_np_load_s", 0.0)),
                        "local_np_load_s": float(acc.get("local_np_load_s", 0.0)),
                        "drive_mount_np_load_s": float(acc.get("drive_mount_np_load_s", 0.0)),
                        "tensor_pack_s": float(acc.get("tensor_pack_s", 0.0)),
                    }
                    yield batch
                finally:
                    _clear_gcs_rewrite(self._dataset)
                    leaf._gcs_staged_root = None
                    self._evict_uri_map(uri_map)
        finally:
            pending = list(futures.values())
            futures.clear()
            for f in pending:
                try:
                    f.cancel()
                except Exception:
                    pass

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
