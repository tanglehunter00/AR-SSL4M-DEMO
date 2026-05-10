#!/usr/bin/env python3
"""
将 pretrain_lists_inventory 目录下的分数据集清单合并为三个训练用列表：

  train_spatial.txt  — 空间：合并「空间」子目录内全部 .txt（不含本脚本生成的 train_*.txt）
  train_semantic.txt — 语义：由 DeepLesion 单行路径聚合成「每行 4 路径逗号分隔」（与 image_dataset 语义分支一致）
  train_contrast.txt — 对比：由 BraTS all_npy.txt 按病例 stem 聚合成「每行 t1n,t1c,t2w,t2f 四路径逗号分隔」

用法:
  python scripts/merge_pretrain_inventory_lists.py --inventory-dir "D:\\path\\pretrain_lists_inventory"
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

MOD_ORDER = ("t1n", "t1c", "t2w", "t2f")


def _read_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def merge_spatial(spatial_dir: Path, out_path: Path) -> int:
    skip = {"train_spatial.txt", "train_semantic.txt", "train_contrast.txt"}
    parts: list[Path] = sorted(spatial_dir.glob("*.txt"))
    parts = [p for p in parts if p.name not in skip]
    all_lines: list[str] = []
    for p in parts:
        all_lines.extend(_read_nonempty_lines(p))
    out_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")
    return len(all_lines)


def merge_semantic(deep_lesion_txt: Path, out_path: Path, seed: int = 42) -> int:
    lines = _read_nonempty_lines(deep_lesion_txt)
    rng = random.Random(seed)
    rng.shuffle(lines)
    n = len(lines) - (len(lines) % 4)
    lines = lines[:n]
    rows = []
    for i in range(0, n, 4):
        rows.append(",".join(lines[i : i + 4]))
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def _bra_ts_group_key(uri: str) -> tuple[str, str] | None:
    """返回 (目录前缀, stem)，stem 不含 .t1n 等后缀。"""
    u = uri.strip()
    if not u.endswith(".npy"):
        return None
    for mod in MOD_ORDER:
        suf = f".{mod}.npy"
        if u.lower().endswith(suf):
            stem_full = u[: -len(suf)]
            parent = stem_full.rsplit("/", 1)[0] if "/" in stem_full else ""
            stem = stem_full.rsplit("/", 1)[-1]
            return (parent, stem)
    return None


def merge_contrast(all_npy_txt: Path, out_path: Path) -> int:
    lines = _read_nonempty_lines(all_npy_txt)
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    skipped = 0
    for ln in lines:
        parsed = _bra_ts_group_key(ln.replace("\\", "/"))
        if not parsed:
            skipped += 1
            continue
        parent, stem = parsed
        mod = None
        lower = ln.lower()
        for m in MOD_ORDER:
            if lower.endswith(f".{m}.npy"):
                mod = m
                break
        if mod:
            groups[(parent, stem)][mod] = ln
    rows: list[str] = []
    for key in sorted(groups.keys()):
        g = groups[key]
        if all(m in g for m in MOD_ORDER):
            rows.append(",".join(g[m] for m in MOD_ORDER))
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    if skipped:
        print(f"[contrast] 跳过无法解析模态的行数: {skipped}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inventory-dir",
        type=Path,
        required=True,
        help="包含「空间」「语义」「对比」子目录的 pretrain_lists_inventory 根目录",
    )
    ap.add_argument("--semantic-seed", type=int, default=42)
    args = ap.parse_args()
    root: Path = args.inventory_dir.resolve()

    spatial_dir = root / "空间"
    semantic_dir = root / "语义"
    contrast_dir = root / "对比"
    deep_txt = semantic_dir / "DeepLesion.txt"
    brats_txt = contrast_dir / "all_npy.txt"

    for name, p in [("空间", spatial_dir), ("语义", semantic_dir), ("对比", contrast_dir)]:
        if not p.is_dir():
            raise SystemExit(f"缺少子目录: {p} （期望存在「{name}」）")

    out_spatial = root / "train_spatial.txt"
    out_semantic = root / "train_semantic.txt"
    out_contrast = root / "train_contrast.txt"

    n_sp = merge_spatial(spatial_dir, out_spatial)
    if not deep_txt.is_file():
        raise SystemExit(f"缺少语义清单: {deep_txt}")
    n_se = merge_semantic(deep_txt, out_semantic, seed=args.semantic_seed)
    if not brats_txt.is_file():
        raise SystemExit(f"缺少对比清单: {brats_txt}")
    n_co = merge_contrast(brats_txt, out_contrast)

    print(f"已写入: {out_spatial}  ({n_sp} 行)")
    print(f"已写入: {out_semantic} ({n_se} 行, 每行 4 路径)")
    print(f"已写入: {out_contrast} ({n_co} 行, 每行 BraTS 四模态)")


if __name__ == "__main__":
    main()
