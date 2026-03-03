# Work Log: Dependency Stabilization & MediaPipe API Compatibility

**Date**: February 10, 2026  
**Type**: Infrastructure / Packaging  
**Status**: ✅ Complete

---

## Problem Statement

The project exhibited critical dependency failures that violated the principle that **successful installation must guarantee successful runtime**:

1. **Editable install succeeded but runtime failed**: `pip install -e .` completed without errors, but core functionality crashed with `AttributeError: module 'mediapipe' has no attribute 'solutions'`
2. **Environment inconsistency**: Conda `gesture` environment had MediaPipe 0.10.32 (broken API), pyenv environment had 0.10.21 (working API)
3. **False validation**: Phase 1-1 validation scripts ran on wrong environment (pyenv), appearing to pass while actual target environment was broken
4. **Fragmented dependency declarations**: Version constraints missing, allowing incompatible versions to be installed

**Root Cause**: MediaPipe 0.10.22+ removed the legacy `mp.solutions.hands` API and replaced it with `mediapipe.tasks.python.vision.HandLandmarker`. The code uses the OLD API, but dependency declarations allowed installation of newer incompatible versions.

---

## Investigation Process

### 1. Environment Discovery

Checked active Python and MediaPipe versions across environments:

```bash
# Pyenv environment (accidentally used for validation)
Python: 3.11.9 (pyenv)
MediaPipe: 0.10.21
Has solutions: True
Has hands: True

# Conda gesture environment (intended target)
Python: 3.10.19 (conda)
MediaPipe: 0.10.32
Has solutions: False  # ❌ BROKEN
Dir: ['Image', 'ImageFormat', 'tasks']  # No solutions attribute
```

**Key Finding**: Conda environment installed MediaPipe 0.10.32 which removed `mp.solutions` API entirely.

### 2. API Archaeology

Investigated MediaPipe version history:

- **v0.10.21 (Feb 8, 2025)**: Last version with `mp.solutions.hands` API
- **v0.10.22 (Mar 18, 2025)**: Removed legacy solutions API, requires migration to `mediapipe.tasks.python.vision.HandLandmarker`
- **v0.10.32 (current)**: Fully removed `mp.solutions`, only `tasks` API available

**Code Impact**:

```python
# Current code (hand_detector.py line 71)
self._mp_hands = mp.solutions.hands  # ❌ Fails in 0.10.22+

# New API (requires code migration)
from mediapipe.tasks.python import vision
hand_landmarker = vision.HandLandmarker.create_from_options(...)
```

### 3. Dependency Audit

Reviewed all dependency declarations:

| File              | Issue                                | Impact                       |
| ----------------- | ------------------------------------ | ---------------------------- |
| `pyproject.toml`  | `mediapipe>=0.10.0` - no upper bound | Allowed incompatible 0.10.32 |
| `environment.yml` | `mediapipe>=0.10.0` - no upper bound | Same issue                   |
| `setup.py`        | Uses pyproject.toml                  | Inherits problem             |

**Transitive Dependency Risk**: MediaPipe 0.10.21 brings in:

- `numpy<2` (explicit constraint)
- `protobuf<5,>=4.25.3` (explicit constraint)
- `opencv-contrib-python` (no version constraint in MediaPipe)

---

## Solution Implemented

### 1. Pinned MediaPipe Version

**File**: `pyproject.toml`

```python
dependencies = [
    "opencv-python>=4.8.0,<5.0.0",       # Camera capture and image operations
    "numpy>=1.24.0,<2.0.0",              # Array operations (compatible with MP 0.10.21)
    "pillow>=9.0.0,<13.0.0",             # Image format support
    "mediapipe==0.10.21",                # Hand detection (last version with solutions API)
]
```

**Rationale**:

- **Exact pin (`==0.10.21`)**: Prevents automatic upgrades to incompatible versions
- **Upper bounds on other deps**: Ensures compatibility with MediaPipe's transitive requirements
- **numpy<2.0.0**: MediaPipe 0.10.21 requires numpy 1.x
- **opencv-python<5.0.0**: Avoid breaking changes in future major version

### 2. Synchronized conda Environment

**File**: `environment.yml`

```yaml
- pip:
    # MediaPipe - hand detection (pin to 0.10.21 - last version with solutions API)
    # Versions 0.10.22+ removed mp.solutions.hands and require migration to tasks.vision API
    - mediapipe==0.10.21
```

**Documentation added**: Inline comment explains WHY the pin exists, preventing future developers from "fixing" it by upgrading.

### 3. Conda-Pip Boundary Clarification

**Established Rule**:

- **Conda manages**: Python runtime + system-level dependencies (OpenCV from conda-forge for better native integration)
- **Pip manages**: Python project-specific packages (MediaPipe only available via pip)

**No duplication**: Removed Python package overlaps between conda and pip sections.

---

## Verification Results

### Test 1: Existing Environment Repair

```bash
# Reinstalled MediaPipe 0.10.21 in conda gesture environment
pip install mediapipe==0.10.21

# Verification
python -c "import mediapipe as mp; ..."
# MediaPipe: 0.10.21
# Has solutions: True
# Has hands: True
# ✅ PASS
```

### Test 2: Editable Install

```bash
pip install -e .

# Result:
# Successfully installed semantic-gesture-system-0.1.0
# All dependencies resolved:
#   - mediapipe==0.10.21 (pinned)
#   - numpy==1.26.4 (within bounds)
#   - opencv-python==4.11.0.86 (within bounds)
#   - pillow==12.1.0 (within bounds)
# ✅ PASS
```

### Test 3: Runtime Validation

```bash
python -c "from input_manager import InputManager; from recognition_engine.hand_detection import HandDetector; ..."

# Output:
# ✓ Imports successful
# ✓ HandDetector initialized
# ✓ MediaPipe Hands loaded
# ✅ PASS
```

### Test 4: Input Layer Validation Script

```bash
python scripts/validate_input_layer.py

# Results:
# camera              : ✓ PASS
# detection           : ✓ PASS (82.7% detection rate, 99.8% avg confidence)
# failure_modes       : ✓ PASS
# ✅ MediaPipe solutions.hands API working
```

### Test 5: Clean Environment Installation

```bash
# Created fresh conda environment
conda create -n gesture-test python=3.10 -y
conda activate gesture-test
pip install -e .

# Verification
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp.solutions, 'hands'))"
# MediaPipe: 0.10.21 True
# ✅ PASS

# Cleanup
conda env remove -n gesture-test -y
```

---

## Migration Path (Future Work)

**Technical Debt**: Code still uses deprecated `mp.solutions` API.

**Phase 2 Migration** (after Phase 1-2 CLIP integration):

1. **Code Migration**: Update `src/recognition_engine/hand_detection/hand_detector.py` to use `mediapipe.tasks.python.vision.HandLandmarker`
2. **Dependency Update**: Change `mediapipe==0.10.21` → `mediapipe>=0.10.22,<1.0.0`
3. **Testing**: Verify detection accuracy equivalent to legacy API
4. **Documentation**: Update architecture docs with new API usage

**API Comparison**:

| Aspect       | Legacy (0.10.21)             | TasksVision (0.10.22+)                        |
| ------------ | ---------------------------- | --------------------------------------------- |
| Import       | `mp.solutions.hands`         | `mediapipe.tasks.python.vision`               |
| Class        | `mp.solutions.hands.Hands()` | `vision.HandLandmarker.create_from_options()` |
| Detection    | `hands.process(frame)`       | `hand_landmarker.detect(mp_image)`            |
| Running Mode | `static_image_mode`          | `RunningMode.IMAGE/VIDEO/LIVE_STREAM`         |

**Estimated Effort**: 2-4 hours (code update + testing)

---

## Lessons Learned

### 1. Successful Install ≠ Successful Environment

**Problem**: `pip install` completed without errors, but runtime crashed immediately.

**Cause**: Dependency solvers only check compatibility at install time, not API compatibility.

**Solution**:

- Pin exact versions for critical dependencies with unstable APIs
- Document WHY pins exist to prevent "cleanup" by future developers
- Always verify runtime after installation

### 2. Version Ranges Must Reflect API Stability

**Problem**: `mediapipe>=0.10.0` assumed semantic versioning, but MediaPipe removed major APIs in minor version.

**Reality**: Not all projects follow semver strictly. Breaking changes can occur in minor versions.

**Solution**:

- Research changelogs before allowing version ranges
- Use exact pins (`==`) for libraries with unstable APIs
- Use bounded ranges (`>=X,<Y`) for stable libraries

### 3. Environment Validation Must Use Target Environment

**Problem**: Validation scripts appeared to pass because they ran on pyenv (wrong environment), not conda (correct environment).

**Cause**: Shell didn't activate conda properly; `conda activate` was ineffective due to shell configuration.

**Solution**:

- Use absolute paths to Python executables: `/path/to/env/bin/python`
- Verify environment at start of validation scripts
- Add environment checks to critical scripts

### 4. Transitive Dependencies Are Not Assumptions

**Problem**: Assumed OpenCV from conda would work with MediaPipe's opencv-contrib-python requirement.

**Reality**: MediaPipe explicitly requires `opencv-contrib-python` via pip, creating potential conflicts.

**Solution**:

- Let MediaPipe manage its own OpenCV dependency via pip
- Conda provides system-level OpenCV for other uses
- Document dependency source (conda vs pip) to avoid confusion

---

## Success Criteria (All Met)

- ✅ Fresh environment installation via `pip install -e .` succeeds
- ✅ MediaPipe `solutions.hands` API exists at runtime
- ✅ Input layer validation passes without errors
- ✅ No undocumented dependencies required
- ✅ Clean test environment installation verified
- ✅ Dependency declarations synchronized (pyproject.toml + environment.yml)
- ✅ Future developers cannot accidentally recreate this failure (pins + documentation)

---

## Files Modified

1. **pyproject.toml**: Added exact version pin for MediaPipe, upper bounds for other dependencies
2. **environment.yml**: Added MediaPipe version pin, documented API compatibility requirement
3. **docs/work-logs/2026-02-10_dependency-stabilization.md**: This document

---

## Next Actions

**Immediate** (Phase 1-2):

- ✅ **UNBLOCKED**: Proceed with CLIP integration using stable dependency base

**Phase 2** (Post-CLIP):

- Migrate to MediaPipe TasksVision API (0.10.22+)
- Update dependency pins to allow newer MediaPipe versions
- Verify detection accuracy with new API

**Ongoing**:

- Monitor MediaPipe releases for API stability
- Review dependency health quarterly
- Add CI/CD checks for dependency compatibility

---

## Validation Status

**Environment Reliability**: ✅ **VALIDATED**  
**Installation Process**: ✅ **REPRODUCIBLE**  
**Runtime Compatibility**: ✅ **VERIFIED**

**Phase 1-1 Status**: ✅ **COMPLETE AND STABLE**

Phase 1-2 (CLIP Integration) is now **UNBLOCKED** and ready to begin.
