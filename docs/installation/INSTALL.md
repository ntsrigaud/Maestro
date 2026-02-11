# Installation Guide

**Updated:** February 12, 2026  
**Status:** Post-packaging refactor - unified namespace

---

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml

# 2. Activate environment
conda activate gesture

# 3. Install package (unified install - no extras needed)
pip install -e .

# 4. Verify installation
python -c "from semantic_hgr.representation.clip_encoder import CLIPEncoder; print('✅ Installation verified')"
```

---

## Prerequisites

- **Conda or Miniconda** ([download](https://docs.conda.io/en/latest/miniconda.html))
- **10 GB disk space** (for packages + CLIP models)
- **Python 3.10+** (automatically installed by environment.yml)
- **Git** (for cloning repository)

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
```

**What this installs:**

- Python 3.10+
- PyTorch (with GPU support if available)
- OpenCV, MediaPipe, NumPy
- Development tools (pytest, ruff)

### 3. Activate Environment

```bash
conda activate gesture
```

### 4. Install Package

```bash
pip install -e .
```

**What `-e` (editable mode) does:**

- Links to source code directory (no copying)
- Code changes immediately available
- No reinstall needed after modifications
- Ideal for development

**What gets installed:**

- All dependencies from `pyproject.toml` (including CLIP)
- `semantic_hgr` package namespace
- 7 console entry points (CLI commands)

**Important:** CLIP dependencies are now **included in base install**. No extras (`[clip]`) required.

---

## Verify Installation

### Test 1: Import Check

```python
from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor
from semantic_hgr.input_manager.input_manager import InputManager

print("✅ All imports successful")
```

### Test 2: Entry Points

```bash
# Test CLI commands
prepare-calibration-dataset --help
calibration-study --help
realtime-demo --help
```

**Available commands:**

- `prepare-calibration-dataset` - Generate calibration dataset
- `calibration-study` - Run confidence calibration
- `intent-overlap-analysis` - Analyze intent overlaps
- `realtime-demo` - Real-time gesture demo
- `validate-clip-integration` - Validate CLIP setup
- `validate-hand-detection` - Validate MediaPipe
- `validate-input-layer` - Validate input layer

### Test 3: Run Unit Tests

```bash
pytest tests/ -v
```

**Expected:** All tests pass (22 encoder tests + 1 semantic test + others)

---

## Namespace and Imports

**Current Namespace:** `semantic_hgr` (PEP 420 compliant)

**Import Pattern:**

```python
from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor
from semantic_hgr.input_manager.input_manager import InputManager
```

**Do NOT use:**

```python
# ❌ OLD (will not work)
from representation.clip_encoder import CLIPEncoder

# ❌ OLD (will not work)
import sys
sys.path.insert(0, 'src')
from representation.clip_encoder import CLIPEncoder
```

---

## Dependencies

### Core Dependencies (from pyproject.toml)

**Vision & ML:**

- `torch` - Deep learning framework
- `torchvision` - Vision utilities
- `opencv-python` - Image processing
- `mediapipe` - Hand detection
- `pillow` - Image loading
- `numpy` - Numerical operations

**CLIP (included in base):**

- `clip @ git+https://github.com/openai/CLIP.git` - OpenAI CLIP
- `ftfy` - Text normalization
- `regex` - Pattern matching

**Development:**

- `pytest` - Testing framework
- `ruff` - Linting and formatting

### Optional Dependencies

**Jupyter (for notebooks):**

```bash
conda install jupyter
```

**GPU Support:**
PyTorch with CUDA automatically installed by conda if GPU detected.

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'semantic_hgr'

**Cause:** Package not installed or wrong namespace

**Solution:**

```bash
# Reinstall in editable mode
pip install -e .

# Verify installation
pip show semantic-gesture-system
```

### Issue: CLIP import fails

**Cause:** CLIP not installed (shouldn't happen with new setup)

**Solution:**

```bash
# CLIP should already be installed, but if not:
pip install -e .  # Reinstall (includes CLIP)
```

### Issue: Entry points not found

**Cause:** Package not properly installed

**Solution:**

```bash
# Reinstall
pip uninstall semantic-gesture-system
pip install -e .

# Verify
which prepare-calibration-dataset
```

### Issue: Import works in terminal but not in IDE

**Cause:** IDE using wrong Python interpreter

**Solution:**

- Point IDE to conda environment Python: `~/miniconda3/envs/gesture/bin/python`
- Restart IDE after changing interpreter

### Issue: Tests fail with import errors

**Cause:** Running tests outside package context

**Solution:**

```bash
# Run from repository root
pytest tests/ -v

# Or install in editable mode first
pip install -e .
```

---

## Development Workflow

### Making Code Changes

1. Edit source files in `src/semantic_hgr/`
2. Changes are immediately available (editable mode)
3. No reinstall needed
4. Run tests to verify: `pytest tests/ -v`

### Adding New Dependencies

1. Add to `pyproject.toml` under `dependencies = [...]`
2. Reinstall: `pip install -e .`
3. Verify: `pip list | grep <package-name>`

### Creating New Entry Points

1. Add to `pyproject.toml` under `[project.scripts]`
2. Reinstall: `pip install -e .`
3. Test: `<command-name> --help`

---

## Packaging Details

**See:** [docs/operations/PACKAGING_REFACTOR_REPORT.md](../operations/PACKAGING_REFACTOR_REPORT.md)

**Key Changes (February 12, 2026):**

- ✅ Unified `semantic_hgr` namespace
- ✅ CLIP moved to base dependencies (no extras)
- ✅ 7 console entry points registered
- ✅ PEP 420 namespace packages
- ✅ All imports use absolute paths

---

## Next Steps

After installation:

1. **Understand the system:** [docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md](../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md)
2. **Check implementation status:** [docs/development/PROJECT_PROGRESS.md](../development/PROJECT_PROGRESS.md)
3. **Run demo:** `realtime-demo`
4. **Review tests:** `pytest tests/ -v`

---

**Installation complete! 🎉**
