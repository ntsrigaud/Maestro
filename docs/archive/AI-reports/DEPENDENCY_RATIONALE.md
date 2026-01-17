# Dependency Rationale & Package Classification

**Last Updated:** January 17, 2026  
**Purpose:** Document why each dependency exists and how it should be installed

---

## Executive Summary

This project uses **73% Conda, 27% pip** for package management.

**Why hybrid?**
- Conda: Native binaries (PyTorch, OpenCV), better platform compatibility
- Pip: Latest APIs (MediaPipe 0.10.x), specialized ML tools

**Total dependencies:**
- Core runtime: 8 packages
- Training/notebooks: 7 packages
- Optional research: 4 packages

---

## Package Classification Matrix

| Package | Module Import | Install Method | Channel/Source | Justification |
|---------|---------------|----------------|----------------|---------------|
| **opencv-python** | `import cv2` | Conda | conda-forge | Native C++ libs, system integration |
| **numpy** | `import numpy` | Conda | conda-forge | Optimized BLAS/LAPACK |
| **mediapipe** | `import mediapipe` | **Pip** | PyPI | 0.10.x unavailable in conda |
| **pytorch** | `import torch` | Conda | pytorch | Platform-optimized binaries (CUDA/Metal) |
| **torchvision** | `import torchvision` | Conda | pytorch | Part of PyTorch ecosystem |
| **torchaudio** | `import torchaudio` | Conda | pytorch | Part of PyTorch ecosystem |
| **timm** | `import timm` | Conda | conda-forge | Available in conda-forge |
| **pillow** | `from PIL import Image` | Conda | conda-forge | Image I/O, stable in conda |
| **scikit-learn** | `import sklearn` | Conda | conda-forge | SciPy stack dependency |
| **pandas** | `import pandas` | Conda | conda-forge | Data science ecosystem |
| **matplotlib** | `import matplotlib` | Conda | conda-forge | Visualization, conda-stable |
| **seaborn** | `import seaborn` | Conda | conda-forge | Statistical plots |
| **tqdm** | `import tqdm` | Conda | conda-forge | Pure Python but in conda |
| **jupyter** | - | Conda | conda-forge | Notebook environment |
| **ipykernel** | - | Conda | conda-forge | Jupyter kernel |
| **sentence-transformers** | `from sentence_transformers import` | **Pip** | PyPI | Not in conda |
| **huggingface-hub** | `from huggingface_hub import` | **Pip** | PyPI | Official distribution on PyPI only |
| **kagglehub** | `import kagglehub` | **Pip** | PyPI | Kaggle-specific, PyPI only |
| **gdown** | `import gdown` | **Pip** | PyPI | Google Drive downloader |

---

## Detailed Justification

### Core Runtime Dependencies

#### 1. opencv-python (Conda)
**Import:** `import cv2`  
**Used in:** All vision modules, demos, InputManager  
**Why Conda?**
- Depends on FFmpeg, system video codecs
- Better integration with Qt/GTK for display
- Conda handles platform-specific binary wheels
- `conda-forge` provides latest stable builds

**Alternative:** `pip install opencv-python` works but may have codec issues

---

#### 2. numpy (Conda)
**Import:** `import numpy as np`  
**Used in:** Everywhere (arrays, image data, tensors)  
**Why Conda?**
- Conda links to optimized BLAS/LAPACK libraries
- Intel MKL acceleration on compatible platforms
- Better performance than pip-installed NumPy

**Evidence:** Standard practice for scientific Python

---

#### 3. mediapipe (Pip) ⚠️
**Import:** `import mediapipe as mp`  
**Used in:** HandDetector (hand landmark detection)  
**Why Pip?**
- **Conda version outdated:** Last conda package is 0.9.3.0 (deprecated API)
- **Project requires:** 0.10.21+ (new Task API)
- **Conda issues:** Binary incompatibilities, broken imports
- **PyPI maintained:** Google officially supports PyPI

**Critical:** This is the ONLY dependency that MUST be pip

**Evidence:**
```bash
# Conda fails:
$ conda search mediapipe
# Shows only 0.9.x versions

# Pip succeeds:
$ pip install mediapipe>=0.10.21
# Installs latest with working API
```

---

#### 4. pytorch, torchvision, torchaudio (Conda)
**Import:** `import torch`, `import torchvision`  
**Used in:** Model training, inference, transforms  
**Why Conda?**
- **Platform-optimized:** CUDA, ROCm, Metal acceleration
- **Binary compatibility:** Matches system drivers
- **Faster installation:** Pre-built wheels from pytorch channel
- **Ecosystem:** torchvision/torchaudio must match torch version

**Alternative:** `pip install torch` works but slower, may miss optimizations

---

#### 5. timm (Conda)
**Import:** `import timm`  
**Used in:** ViT-B/16 model loading (`timm.create_model('vit_base_patch16_224')`)  
**Why Conda?**
- Available in conda-forge with stable versions
- No native dependencies, but conda ensures compatibility

**Alternative:** `pip install timm` equally valid

---

#### 6. pillow (Conda)
**Import:** `from PIL import Image`  
**Used in:** Image loading, transforms, preprocessing  
**Why Conda?**
- Depends on libjpeg, libtiff, zlib
- Conda manages these C libraries automatically
- Better cross-platform consistency

---

### Training & Notebook Dependencies

#### 7. scikit-learn (Conda)
**Import:** `from sklearn.metrics import ...`  
**Used in:** Confusion matrices, accuracy metrics, classification reports  
**Why Conda?**
- Part of SciPy stack (depends on scipy, numpy)
- Conda resolves complex dependency graph
- Better numerical library linking

---

#### 8. pandas (Conda)
**Import:** `import pandas as pd`  
**Used in:** Dataset organization scripts, HaGRID download  
**Why Conda?**
- NumPy/SciPy integration
- C extensions benefit from conda's compiler

---

#### 9. matplotlib, seaborn (Conda)
**Import:** `import matplotlib.pyplot as plt`, `import seaborn as sns`  
**Used in:** Training plots, confusion matrices, visualizations  
**Why Conda?**
- GUI backends (Tk, Qt) easier to configure in conda
- Font rendering dependencies handled by conda

---

#### 10. tqdm (Conda)
**Import:** `from tqdm import tqdm`  
**Used in:** Progress bars (training, dataset downloads)  
**Why Conda?**
- Available in conda-forge
- No strong reason, could be pip

**Alternative:** `pip install tqdm` equally valid

---

#### 11. jupyter, ipykernel (Conda)
**Used in:** Notebook execution environment  
**Why Conda?**
- Notebook ecosystem traditionally conda-based
- Conda handles kernel registration automatically
- Better environment isolation

---

### Optional Research Dependencies

#### 12. sentence-transformers (Pip) ⚠️
**Import:** `from sentence_transformers import SentenceTransformer`  
**Used in:** Semantic gesture learning (experimental)  
**Why Pip?**
- **Not in conda:** No conda-forge package available
- **Actively maintained on PyPI:** Monthly updates
- **Transitive deps:** Requires transformers, tokenizers (also pip-only)

**Install:** `pip install -e ".[semantic]"`

---

#### 13. huggingface-hub (Pip) ⚠️
**Import:** `from huggingface_hub import HfApi, hf_hub_download`  
**Used in:** Model upload/download scripts (optional)  
**Why Pip?**
- **Official distribution:** Hugging Face only supports PyPI
- **No conda package:** Not available in any conda channel

**Install:** `pip install -e ".[huggingface]"`

---

#### 14. kagglehub, gdown (Pip) ⚠️
**Import:** `import kagglehub`, `import gdown`  
**Used in:** Dataset download scripts (optional)  
**Why Pip?**
- **Specialized tools:** Not suitable for conda packaging
- **PyPI-only:** No conda equivalents

**Install:** `pip install -e ".[datasets]"`

---

## Version Pinning Strategy

### Minimum Version (>=)
Used for: Core libraries where we need a feature introduced in a specific version

```yaml
pytorch>=2.0.0       # Need torch.mps support (Metal)
numpy>=1.24.0        # Need specific dtype features
mediapipe>=0.10.21   # Need 0.10.x Task API (0.9.x incompatible)
```

**Rationale:** Allows security updates while ensuring compatibility

---

### Exact Version (==)
Used for: NONE currently

**Rationale:** Only pin exact versions when known incompatibilities exist

**Future:** If MediaPipe 0.11.x breaks API, use `mediapipe==0.10.21`

---

### No Pin
Used for: Standard library, utilities

```yaml
tqdm        # No breaking changes expected
pandas      # Stable API
```

**Rationale:** Maximize compatibility with user's existing environment

---

## Platform-Specific Considerations

### macOS (Apple Silicon)

**PyTorch Metal Acceleration:**
```yaml
pytorch::pytorch>=2.0.0  # Includes Metal Performance Shaders (MPS)
```

**OpenCV:** Conda builds include Apple-specific optimizations

**MediaPipe:** Pip wheel includes ARM64 binaries

---

### Linux (CUDA)

**PyTorch with CUDA:**
```bash
# Override environment.yml:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**OpenCV:** May need system dependencies:
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

---

### Windows

**Visual C++ Runtime:** Required by PyTorch, auto-installed by conda

**Long paths:** Enable before cloning:
```powershell
git config --system core.longpaths true
```

---

## Transitive Dependencies

These are installed automatically:

| Package | Brings In | Managed By |
|---------|-----------|------------|
| **pytorch** | numpy, sympy, typing-extensions | Conda |
| **torchvision** | pytorch, pillow | Conda |
| **mediapipe** | protobuf, opencv-python, numpy | Pip |
| **sentence-transformers** | transformers, torch, tokenizers | Pip |
| **matplotlib** | cycler, kiwisolver, pyparsing | Conda |
| **scikit-learn** | scipy, threadpoolctl | Conda |

**No action needed:** Dependencies resolve automatically

---

## Conflict Resolution

### Known Conflict: opencv-python vs opencv-contrib-python

**Symptom:** Both installed, import cv2 fails

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless
conda install -c conda-forge opencv
```

---

### Known Conflict: PyTorch symbol errors

**Symptom:** `ImportError: undefined symbol: iJIT_NotifyEvent`

**Solution:**
```bash
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio -c pytorch
```

**Cause:** Mixing conda and pip PyTorch builds

---

## Validation Checklist

After environment setup, verify:

```bash
# 1. Core imports work
python -c "import cv2, numpy, torch, mediapipe, timm"

# 2. No duplicate OpenCV
pip list | grep opencv
# Should show ONLY: opencv-python (from conda)

# 3. MediaPipe version correct
python -c "import mediapipe as mp; print(mp.__version__)"
# Should show: 0.10.21 or higher

# 4. PyTorch accelerator available
python -c "import torch; print(torch.backends.mps.is_available())"
# macOS: True, Linux: False (use cuda.is_available())

# 5. Project package installed
python -c "from input_manager import InputManager"
# Should succeed without error
```

---

## Future Considerations

### Can requirements.txt be eliminated?

**Current:** No, pip packages (MediaPipe, sentence-transformers) required

**Future:** If MediaPipe releases conda builds for 0.10.x, can move to conda-only

### Docker alternative?

For maximum reproducibility:
```dockerfile
FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "gesture", "/bin/bash", "-c"]
RUN pip install -e .
```

Not implemented; conda setup is sufficient for now

---

## References

- PyTorch Installation Guide: https://pytorch.org/get-started/locally/
- MediaPipe 0.10.x Migration: See `HAND_DETECTOR_REFACTORING.md`
- Conda vs Pip: https://www.anaconda.com/blog/understanding-conda-and-pip

---

**Summary:** This project requires a conda + pip hybrid approach. 19 dependencies total, 15 from conda, 4 from pip. MediaPipe is the critical pip dependency due to API version requirements.
