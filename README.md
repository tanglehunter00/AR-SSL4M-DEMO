# GASP-AR3D · Extended 3D Autoregressive Medical SSL

Research codebase extending **Autoregressive Sequence Modeling for 3D Medical Image Representation** (Wang et al., AAAI 2025). The baseline formulation follows the [official implementation](https://github.com/serena9525/AR-SSL4M); this repository adds geometric–sparse attention ideas on top (`newFullPretrain`).

---

## Baseline (original paper & code)

**Paper:** *Autoregressive Sequence Modeling for 3D Medical Image Representation* — [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32848) · [arXiv:2409.08691](https://arxiv.org/abs/2409.08691)

**Idea (short):**

- Turn 3D CT/MRI into **patch / visual-token sequences** with **spatial**, **contrast**, and **semantic** construction rules.
- Train a **causal decoder** with **autoregressive prediction** (next-token / patch reconstruction) under **prefix-style attention** and a **random startup** strategy for robustness.
- Fine-tune on segmentation and classification tasks.

**Code alignment:** `pretrain/` matches the upstream training layout (FSDP, configs, dataset lists). Upstream notes and dataset links are preserved below.

---

## What we add (this repo)

Implemented primarily under **`newFullPretrain/`** (see `newNewModel.py`):

| Component | Role |
|-----------|------|
| **Geometry bias** | Per-head decayed bias on attention scores; distance on the **3D patch index grid** uses **Manhattan (L1)** distance between `(t, h, w)` indices. Decay coefficients follow a **D-Former v2–style** construction (see code comments). |
| **Hybrid sparse attention** | Optional **local 3D window + axial dilated / sparse** mask, combined with the existing causal mask, to reduce full dense attention cost on long token sequences. |

Default volume configuration in `newFullPretrain/configs/datasets.py` is **`img_size = 128³`**, **`patch_size = 16³`**, so the inner patch grid is **`8×8×8` tokens** (plus start/end tokens in the full sequence). Change `img_size` / `patch_size` if you need a different grid (e.g. `32³` patches requires a matching resolution / patch ratio).

Additional training utilities (e.g. BraTS-oriented loaders / hybrid dataset) live next to `main.py` for large-scale contrast pretraining; enable paths follow `train_config` and `dataset_config` in code.

---

## Repository layout

| Path | Description |
|------|-------------|
| `pretrain/` | Baseline-style pretraining (upstream-aligned). |
| `newFullPretrain/` | **Main extended pretraining** (geometry bias + hybrid sparse attention, etc.). |
| `newPretrain/` | Earlier experiment branch; kept for reference. |
| `preprocess/` | Scripts to build spatial / contrast / semantic sub-volumes and training lists. |
| `downstream/` | Segmentation (MSD, LA), nodule classification (MedMNIST), COVID classification (RICORD). |
| `*.ipynb` | Notebooks for data download or ad-hoc workflows. |

---

## Environment

Recommended (from upstream):

- CUDA 11.8  
- Python 3.8  
- `torch==2.1.2`  
- `transformers==4.37.0.dev0`  
- `monai==1.3.0`  

Install any extra dependencies your chosen entrypoint imports (e.g. `fire`, `timm`).

---

## Data

Same spirit as the original release:

**Pretrain (three sequence types):**

- **Spatial:** RibFrac, TCIA COVID-19, AMOS22, ISLES 2022, AbdomenCT-1K, TotalSegmentator, Verse 2020, RSNA-2022-CSFD, RSNA-2020-PED, STOIC, FLARE22/23, …  
- **Contrast:** [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/files/) (GLI, MEN, MET, PED, SSA).  
- **Semantic:** [DeepLesion](https://nihcc.app.box.com/v/DeepLesion).

**Fine-tuning:** MSD, LA, Lung nodule (e.g. Zenodo nodulemnist3d), RICORD-1A/1B — see original paper supplement and links in the **Pre-processing** section below.

---

## Pre-processing

- **Spatial:** `preprocess/proc_spatial_sequence.py`  
- **Contrast:** `preprocess/proc_contrast_sequence.py`  
- **Semantic:** `preprocess/proc_semantic_sequence.py`  
- **Training file lists:** `preprocess/gen_pretrain_list.py`  

**Fine-tuning prep:** `preprocess/preprocess_npz.py` (nodule), `preprocess/dicom2nifti.py` + `preprocess/preprocess_nii.py` (COVID), etc.

---

## Pre-training

**Baseline stack:**

```bash
cd pretrain
# edit run_scripts/pretrain.sh then:
bash run_scripts/pretrain.sh
```

**Extended stack (this project’s focus):**

```bash
cd newFullPretrain
# edit run_scripts/pretrain.sh (paths, GPUs, FSDP flags) then:
bash run_scripts/pretrain.sh
```

Entry point: `newFullPretrain/main.py` (Fire / CLI flags mirror the baseline config pattern).

**Original released checkpoint (paper authors):** [Google Drive](https://drive.google.com/file/d/1pJRaE9H4C2oc_NiMFA2XqTAE3BGtBJvp/view?usp=drive_link)

---

## Fine-tuning (downstream)

Scripts under `downstream/*/run_scripts/`:

- Segmentation: `downstream/segmentation/run_scripts/run_ssl.sh`  
- Lung nodule: `downstream/nodule/run_scripts/run_ssl.sh`  
- COVID: `downstream/COVID/run_scripts/run_ssl.sh`  

Downstream code is largely inherited from the baseline release; adapt checkpoints and configs to your own pretraining run.

---

## Citation

If you use the **original method**, cite the paper:

```bibtex
@inproceedings{wang2025autoregressive,
  title={Autoregressive Sequence Modeling for 3D Medical Image Representation},
  author={Wang, Siwen and Wang, Churan and Gao, Fei and Su, Lixian and Zhang, Fandong and Wang, Yizhou and Yu, Yizhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={7871--7879},
  year={2025}
}
```

This repository is a **derivative** of [AR-SSL4M](https://github.com/serena9525/AR-SSL4M); please also credit that codebase if you reuse its training or data pipeline.

---

## Acknowledgments

Implementation builds on the authors’ release and, as noted there, on ideas and code from projects such as [Llama Cookbook](https://github.com/meta-llama/llama-cookbook/), [MONAI](https://github.com/Project-MONAI/MONAI), [MedMNIST](https://github.com/MedMNIST/experiments), and [MedCoSS](https://github.com/yeerwen/MedCoSS). Geometry-bias decay follows a pattern described in **D-Former v2** (see comments in `newFullPretrain/newNewModel.py`).
