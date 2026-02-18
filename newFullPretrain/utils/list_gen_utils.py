"""
Utilities for generating pretrain list files.
Supports: LIDC/spatial (loose npy), BraTS contrast (tar.gz without full extract), DeepLesion semantic.
"""
import os
import tarfile
import io


def gen_spatial_list(patch_dirs, save_path):
    """
    Generate spatial list from loose .npy directories.
    Each line = one absolute path to 128^3 npy file.
    patch_dirs: list of (base_path, dir_name) e.g. [('/content/drive/.../LIDC-IDRI', 'patch_random_spatial'), (..., 'patch_random_lidc')]
    Or list of full dir paths as strings.
    """
    data_list = []
    for item in patch_dirs:
        if isinstance(item, (list, tuple)):
            base_path, dir_name = item
            full_dir = os.path.join(base_path, dir_name)
        else:
            full_dir = item
        if not os.path.exists(full_dir):
            continue
        for f in os.listdir(full_dir):
            if f.endswith('.npy'):
                data_list.append(os.path.join(full_dir, f))
    if data_list:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(data_list))
    return len(data_list)


def gen_contrast_list_from_tar(tar_root, save_path):
    """
    Generate BraTS contrast list from tar.gz files WITHOUT extracting.
    Uses tarfile.getnames() to list members - no disk extraction.
    List format: tar_path:member_base (e.g. /path/to/case.tar.gz:case_id/ds_case_0)
    Dataset will load 4 npy (t1n,t1c,t2w,t2f) from tar at runtime.
    """
    lines = []
    tar_files = []
    for root, _, files in os.walk(tar_root):
        for f in files:
            if f.endswith('.tar.gz'):
                tar_files.append(os.path.join(root, f))

    for tar_path in tar_files:
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                names = tar.getnames()
                # Group by base: xxx.t1n.npy -> base = xxx
                t1n_bases = set()
                for n in names:
                    if n.endswith('.t1n.npy'):
                        base = n[:-len('.t1n.npy')]
                        # Verify all 4 exist
                        t1c = base + '.t1c.npy'
                        t2w = base + '.t2w.npy'
                        t2f = base + '.t2f.npy'
                        if t1c in names and t2w in names and t2f in names:
                            t1n_bases.add(base)
                for base in sorted(t1n_bases):
                    lines.append(f"{tar_path}:{base}")
        except Exception as e:
            print(f"Warning: skip {tar_path}: {e}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    return len(lines)


def gen_semantic_list(npy_dir, save_path, samples_per_category=20000):
    """
    Generate DeepLesion semantic list.
    Each line = 4 comma-separated paths (same lesion type 1-8).
    """
    import random
    random.seed(0)
    all_data_list = []
    if not os.path.exists(npy_dir):
        return 0
    for num in range(8):
        data_list = [
            os.path.join(npy_dir, x) for x in os.listdir(npy_dir)
            if x.endswith(f'_{num + 1}.npy')
        ]
        n_samples = min(samples_per_category, len(data_list) // 4) if len(data_list) >= 4 else 0
        for _ in range(n_samples):
            choose_list = random.sample(data_list, 4)
            all_data_list.append(','.join(choose_list))
    if all_data_list:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(all_data_list))
    return len(all_data_list)


def load_npy_from_tar(tar_path, member_path):
    """Load a single .npy from tar without extracting. Returns numpy array."""
    import numpy as np
    with tarfile.open(tar_path, 'r:gz') as tar:
        f = tar.extractfile(member_path)
        if f is None:
            raise FileNotFoundError(f"Member {member_path} not in {tar_path}")
        return np.load(io.BytesIO(f.read()))
