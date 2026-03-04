# Installation Guide — Architecture A

**Version:** 0.2.0  
**Updated:** 2026  
**Model:** Dual-Stream Supervised (GCN + ViT-B/16)

---

## Quick Start

```bash
# 1. Create conda environment (Python 3.10, PyTorch, timm, torch-geometric)
conda env create -f environment.yml

# 2. Activate environment
conda activate gesture

# 3. Install package in editable mode
pip install -e .

# 4. Verify installation
python scripts/verify_install.py
```

Expected output: `Architecture A dependencies: ALL OK`

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Conda / Miniconda | [download](https://docs.conda.io/en/latest/miniconda.html) |
| Python 3.10 | Automatically installed via `environment.yml` |
| Git | For cloning the repository |
| ~5 GB disk | For packages + HaGRID dataset |
| Apple M1/M2 or CUDA GPU | MPS / CUDA acceleration optional; CPU works |

---

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd Semantic-HGR-System
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate gesture
```

This installs:

- Python 3.10
- PyTorch ≥ 2.0 (pytorch channel)
- torchvision ≥ 0.15
- OpenCV, NumPy, MediaPipe
- scikit-learn, matplotlib, seaborn, pandas
- `timm ≥ 0.9.0` (ViT-B/16 appearance stream)
- `torch-geometric ≥ 2.3.0` (GCN geometry stream)
- pytest, ruff (development tools)

### 3. Install Package

```bash
pip install -e .
```

`-e` (editable mode) links to the source directory — code changes are immediately
reflected without reinstalling.

**Installed console entry points:**

| Command | Module |
|---------|--------|
| `validate-hand-detection` | `semantic_hgr.scripts.validate_hand_detection` |
| `validate-input-layer` | `semantic_hgr.scripts.validate_input_layer` |

### 4. Verify Installation

```bash
python scripts/verify_install.py
```

The script checks:

1. `torch`, `numpy`, `scikit-learn`
2. `timm` import + ViT-B/16 instantiation (86M params, 8 classes)
3. `torch_geometric` import + `GCNConv(3, 64)` instantiation
4. `mediapipe`, `opencv-python`, `HandDetector` import
5. `pandas`, `matplotlib`, `seaborn`

---

## Download Dataset

```bash
make download-hagrid
```

Downloads the HaGRID 8-class subset into `data/hagrid_subset/`:

```
data/hagrid_subset/
    call_gesture/    thumbs_up/
    closed_fist/     thumbs_down/
    ok_sign/         peace_sign/
    open_palm/       pointing/
```

---

## Pre-Training Data Pipeline

```bash
# Step 1 — Offline keypoint extraction
make extract-keypoints
# Reads: data/hagrid_subset/  Writes: data/keypoints/  data/splits/

# Step 2 — Validate keypoints (pre-training gate, Section 2.6)
make validate-keypoints
```

---

## Training

```bash
make train-gcn     # Phase 1: GCN geometry stream  (target ≥65% val Top-1)
make train-rgb     # Phase 2: ViT-B/16 RGB stream  (target ≥88% val Top-1)
make train-dual    # Phase 3+4: Cross-attention fusion + end-to-end tune
```

Config files: `configs/arch_a_gcn.yaml`, `configs/arch_a_rgb.yaml`, `configs/arch_a_dual.yaml`

---

## Evaluation

```bash
make eval
# Writes: results/static_eval_report.json
```

---

## Troubleshooting

### `torch_geometric` installation issues

```bash
pip install torch-geometric
# If scatter/sparse extensions needed:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

### `mediapipe` on Apple Silicon

```bash
pip install mediapipe==0.10.21
```

---

## Development

```bash
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
make test          # pytest tests/
```
