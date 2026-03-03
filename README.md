# Semantic Gesture Understanding System

**A vision-language system that maps hand gestures to semantic intent through CLIP-based visual encoding.**

---

## Project Status

✅ **Phase 1-2 Complete** - Core semantic pipeline (Layers 1-3) operational  
📋 **Phase 2 Next** - Command resolution and execution layers (future work)

**Current Capabilities:**

- Camera input & MediaPipe hand detection
- CLIP-based visual encoding (gesture → 512D embeddings)
- Semantic intent extraction (19 intent vocabulary)
- Probability validation & calibration tools
- Real-time gesture recognition demo

**Documentation:** [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md) | [Installation Guide](docs/installation/INSTALL.md)

---

## What This Project Is

This project implements a **semantic gesture understanding system** that uses vision-language models to extract intent from hand gestures:

1. **Captures gesture data** from camera input (InputManager)
2. **Detects and localizes hands** using MediaPipe (HandDetector)
3. **Extracts semantic intent** using CLIP visual encoder (CLIPEncoder)
4. **Validates probability distributions** with temperature scaling

### Core Objective

> Map hand gestures to semantic intent through **vision-language understanding**, NOT direct gesture-to-command mappings.

Gestures are **expressions of intent**. The system understands meaning through semantic similarity, not just visual patterns.

---

## What This Project Is NOT

- ❌ A gesture classifier (single-label predictions without semantic grounding)
- ❌ A direct gesture-to-command mapper (hard-coded action lookups)
- ❌ A fully autonomous control system (command execution not implemented yet)
- ❌ A production-ready application (research prototype for semantic HGR)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│         SEMANTIC GESTURE UNDERSTANDING SYSTEM                   │
│           (3 Implemented Layers + 2 Future Layers)              │
└─────────────────────────────────────────────────────────────────┘

Camera Input
     ↓
┌────────────────────────────────┐
│ LAYER 1: Perception            │  ✅ IMPLEMENTED
│ • InputManager (camera)        │  • Camera capture (30 FPS)
│ • HandDetector (MediaPipe)     │  • Hand detection & cropping
│ • HandCropper (ROI extraction) │  • 224×224 normalization
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ LAYER 2: Representation        │  ✅ IMPLEMENTED
│ • CLIPEncoder                  │  • CLIP ViT-B/32 visual encoder
│                                │  • Gesture → 512D embeddings
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ LAYER 3: Semantic Intent       │  ✅ IMPLEMENTED
│ • SemanticIntentExtractor      │  • 19-intent vocabulary
│                                │  • Temperature-scaled softmax
│                                │  • Probability validation (10 tests)
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ LAYER 4: Command Resolution    │  📋 FUTURE WORK
│ • SentenceTransformers         │  (Not yet implemented)
│ • Semantic similarity matching │
└────────┬───────────────────────┘
         ↓
┌────────────────────────────────┐
│ LAYER 5: Execution             │  📋 FUTURE WORK
│ • OS command safety            │  (Not yet implemented)
└────────────────────────────────┘
```

**Status Legend:**

- ✅ **IMPLEMENTED** - Code exists, tests pass, production-ready
- 📋 **FUTURE WORK** - Planned but not started

---

## Key Features

### Semantic Understanding (Not Classification)

---

## Key Features

### Semantic Understanding (Not Classification)

**Traditional Approach** (What we DON'T do):

```
Gesture → Classifier → Label → Hard-coded action
```

**Our Approach** (Semantic understanding through CLIP):

```
Gesture → CLIP Visual Encoder → 512D Embedding → Semantic Intent (19 classes)
```

### Implemented Capabilities

The system currently provides:

1. **Real-Time Hand Detection**
   - MediaPipe Hands integration
   - 224×224 ROI extraction
   - 30 FPS camera capture

2. **CLIP-Based Visual Encoding**
   - ViT-B/32 architecture
   - 512-dimensional embeddings
   - Pre-trained vision-language model

3. **Semantic Intent Extraction**
   - 19-intent vocabulary (pointing, grabbing, waving, etc.)
   - Temperature-scaled softmax (τ = 0.05)
   - Probability validation (all ∈ [0,1], sum = 1.0)

4. **Production Tools** (7 console scripts)
   - Calibration study (ECE/MCE/Brier scores)
   - Intent overlap analysis
   - Real-time demo (<120ms latency)
   - Validation scripts (CLIP, hand detection, input layer)

### Research-Backed Design

Every architectural decision grounded in research:

- **CLIP** (Radford et al., 2021) - Vision-language joint embedding
- **MediaPipe Hands** (Zhang et al., 2020) - Real-time hand detection
- **Temperature Scaling** (Guo et al., 2017) - Probability calibration

See [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md) for complete research foundation.

---

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate gesture

# Install package (includes CLIP + MediaPipe)
pip install -e .

# Verify installation
python scripts/verify_install.py
```

**Installation time:** ~5 minutes (CLIP models download automatically on first use)

See [docs/installation/INSTALL.md](docs/installation/INSTALL.md) for detailed setup instructions.

### Try the Demo

```bash
# Real-time gesture recognition (requires webcam)
realtime-demo

# Hand detection validation
validate-hand-detection

# CLIP integration test
validate-clip-integration
```

**Expected output:** Live video feed with hand detection boxes, semantic intent predictions, and confidence scores.

---

## Implementation Status

### ✅ Layers 1-3: COMPLETE & TESTED

**Layer 1: Perception**

- ✅ `InputManager` - Camera capture, frame buffering
- ✅ `HandDetector` - MediaPipe hand detection
- ✅ `HandCropper` - ROI extraction & normalization
- ✅ Tests: `validate-input-layer`, `validate-hand-detection`

**Layer 2: Representation**

- ✅ `CLIPEncoder` - Visual encoding (ViT-B/32)
- ✅ 512D embeddings
- ✅ Tests: 22 encoder tests passing

**Layer 3: Semantic Intent**

- ✅ `SemanticIntentExtractor` - Intent prediction
- ✅ 19-intent vocabulary
- ✅ Probability validation (10 tests)
- ✅ Temperature scaling (τ = 0.05)
- ✅ Tests: 22 semantic tests passing

**Production Tools (7 console scripts):**

- ✅ `prepare-calibration-dataset` - Collect calibration data
- ✅ `calibration-study` - ECE/MCE/Brier analysis
- ✅ `intent-overlap-analysis` - Semantic similarity matrix
- ✅ `realtime-demo` - Live gesture recognition
- ✅ `validate-clip-integration` - CLIP pipeline test
- ✅ `validate-hand-detection` - MediaPipe test
- ✅ `validate-input-layer` - Camera test

**Total Test Coverage:** 54/54 tests passing

**Documentation:**

- ✅ [CLIP Mathematical Specification](docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md)
- ✅ [Production Finalization Guide](docs/operations/PRODUCTION_FINALIZATION_GUIDE.md)
- ✅ [System Architecture](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)

### 📋 Layers 4-5: NOT IMPLEMENTED (Future Work)

**Layer 4: Command Resolution**

- ❌ SentenceTransformers integration
- ❌ Text command vocabulary
- ❌ Semantic similarity matching

**Layer 5: Execution**

- ❌ OS command execution
- ❌ Safety boundaries
- ❌ Action mapping

**Status:** These layers are planned for Phase 2 but not yet started. The current system provides semantic intent extraction only.

---

## Repository Structure

```
Semantic-HGR-System/
├── README.md                              # Project overview
├── pyproject.toml                         # Package configuration
├── environment.yml                        # Conda environment
│
├── docs/
│   ├── README.md                          # Documentation index
│   ├── SEMANTIC_SYSTEM_ARCHITECTURE.md    # ⭐ Authoritative architecture
│   ├── installation/
│   │   └── INSTALL.md                     # ⭐ Installation guide
│   ├── architecture/
│   │   └── CLIP_MATHEMATICAL_SPECIFICATION.md  # Mathematical formulation
│   ├── operations/
│   │   └── PRODUCTION_FINALIZATION_GUIDE.md    # Calibration & validation
│   ├── work-logs/                         # Implementation logs (Feb 2026)
│   └── archive/                           # 🗄️ Archived documentation
│
├── src/semantic_hgr/                      # ✅ Main package
│   ├── input_manager/                     # ✅ Camera capture (Layer 1)
│   │   └── input_manager.py              # InputManager class
│   ├── recognition_engine/
│   │   └── hand_detection/                # ✅ Hand detection (Layer 1)
│   │       ├── hand_detector.py          # MediaPipe integration
│   │       ├── hand_cropper.py           # ROI extraction
│   │       └── hand_data.py              # Data structures
│   ├── representation/                    # ✅ Visual encoding (Layer 2)
│   │   └── clip_encoder.py               # CLIPEncoder class
│   ├── semantics/                         # ✅ Intent extraction (Layer 3)
│   │   └── clip_intent.py                # SemanticIntentExtractor class
│   ├── shared/                            # ✅ Utilities
│   │   └── logger.py                     # Logging configuration
│   └── scripts/                           # ✅ 7 console entry points
│       ├── prepare_calibration_dataset.py
│       ├── calibration_study.py
│       ├── intent_overlap_analysis.py
│       ├── realtime_demo.py
│       ├── validate_clip_integration.py
│       ├── validate_hand_detection.py
│       └── validate_input_layer.py
│
├── tests/                                 # ✅ 54 tests (all passing)
│   ├── test_clip_encoder.py              # 22 encoder tests
│   ├── test_semantic_intent.py           # 22 semantic tests
│   └── test_probability_validation.py    # 10 probability tests
│
├── scripts/                               # Legacy validation scripts
│   ├── verify_install.py
│   └── validate_*.py
│
└── notebooks/                             # Experiments & analysis
    ├── training_results/
    └── archive/
```

**Package Structure:**

- **Namespace:** `semantic_hgr`
- **Entry Points:** 7 console commands (registered in pyproject.toml)
- **Import Pattern:** `from semantic_hgr.representation.clip_encoder import CLIPEncoder`

**Legend:**

- ⭐ Authoritative/critical documentation
- ✅ Implemented and tested
- 🗄️ Archived (historical reference only)

---

## Technology Stack

### Implemented (Production Ready)

| Component            | Technology      | Purpose                  | Status         |
| -------------------- | --------------- | ------------------------ | -------------- |
| **Hand Detection**   | MediaPipe Hands | Locate hands in frame    | ✅ Implemented |
| **Visual Encoding**  | CLIP ViT-B/32   | Gesture → 512D embedding | ✅ Implemented |
| **Semantic Intent**  | CLIP Text Enc.  | Intent classification    | ✅ Implemented |
| **Probability**      | Softmax + Temp  | Calibrated confidence    | ✅ Implemented |
| **Calibration**      | ECE/MCE/Brier   | Reliability metrics      | ✅ Tools ready |
| **Overlap Analysis** | Cosine Sim.     | Intent confusion matrix  | ✅ Tools ready |
| **Demo Framework**   | Real-time CV    | Live gesture recognition | ✅ Implemented |

**Dependencies:** torch, open-clip-torch, mediapipe, opencv-python, scikit-learn, matplotlib

### Future (Phase 2 - Not Started)

| Component              | Technology           | Purpose             | Status     |
| ---------------------- | -------------------- | ------------------- | ---------- |
| **Command Matching**   | SentenceTransformers | Text similarity     | 📋 Planned |
| **Dynamic Gestures**   | VideoCLIP/X-CLIP     | Temporal modeling   | 📋 Planned |
| **Action Recognition** | ActionCLIP           | Video understanding | 📋 Planned |
| **OS Execution**       | PyAutoGUI/Scripting  | Command execution   | 📋 Planned |

---

## Development Guide

### Getting Started

1. **Read Architecture:** [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
2. **Install:** Follow [docs/installation/INSTALL.md](docs/installation/INSTALL.md)
3. **Run Demo:** `realtime-demo` (requires webcam)
4. **Run Tests:** `pytest tests/ -v` (54 tests should pass)

### Code Examples

**Using the CLIP Encoder:**

```python
from semantic_hgr.representation.clip_encoder import CLIPEncoder
import numpy as np

# Initialize encoder
encoder = CLIPEncoder()

# Encode hand gesture image (224×224 RGB)
hand_image = np.random.rand(224, 224, 3).astype(np.float32)
embedding = encoder.encode_image(hand_image)  # Returns 512D vector

print(f"Embedding shape: {embedding.shape}")  # (512,)
```

**Using the Semantic Intent Extractor:**

```python
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor

# Initialize extractor (loads CLIP + intent vocabulary)
extractor = SemanticIntentExtractor()

# Extract intent from hand image
intent, confidence = extractor.predict(hand_image)

print(f"Intent: {intent}, Confidence: {confidence:.3f}")
# Example output: Intent: pointing, Confidence: 0.847
```

**Running Calibration Study:**

```bash
# Prepare calibration dataset (100 samples per intent)
prepare-calibration-dataset --output data/calibration/

# Run calibration analysis
calibration-study --dataset data/calibration/ --output results/

# Output: ECE, MCE, Brier scores + reliability diagrams
```

### Documentation Standards

- **Architecture:** [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md) (authoritative)
- **Operations:** [docs/operations/](docs/operations/) (calibration, validation, deployment)
- **Work Logs:** [docs/work-logs/](docs/work-logs/) (implementation progress)

---

## Validation & Testing

### Test Coverage (54/54 Passing)

**CLIP Encoder Tests (22 tests):**

- Model initialization & device placement
- Image encoding (single + batch)
- Text encoding (single + batch)
- Embedding dimensions & normalization
- Error handling (invalid inputs)

**Semantic Intent Tests (22 tests):**

- Intent prediction accuracy
- Confidence score validation
- 19-intent vocabulary coverage
- Temperature scaling effects
- Edge cases (blank images, invalid inputs)

**Probability Validation Tests (10 tests):**

- Softmax normalization (sum = 1.0)
- Confidence range constraints ([0, 1])
- Temperature scaling behavior
- Logit scaling verification

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_clip_encoder.py -v

# With coverage
pytest tests/ --cov=semantic_hgr --cov-report=html
```

### Performance Benchmarks

**Real-Time Demo (realtime-demo):**

- Hand detection: ~30 FPS (MediaPipe)
- CLIP encoding: ~15 FPS (CPU) / ~60 FPS (GPU)
- End-to-end latency: <120ms (webcam → intent prediction)

**Calibration Results (calibration-study):**

- Expected Calibration Error (ECE): <0.05
- Maximum Calibration Error (MCE): <0.15
- Brier Score: <0.20

See [docs/operations/PRODUCTION_FINALIZATION_GUIDE.md](docs/operations/PRODUCTION_FINALIZATION_GUIDE.md) for detailed metrics.

---

---

## Contributing

This is a research prototype for academic evaluation. The system demonstrates semantic gesture understanding through CLIP-based visual encoding but is not production-ready software.

### Research Context

**Objective:** Demonstrate vision-language models can extract semantic intent from hand gestures without task-specific training.

**Current Status:**

- ✅ Layers 1-3 implemented (perception → representation → semantic intent)
- ✅ 54 tests passing, calibration tools operational
- ❌ Command resolution (Layer 4) not implemented
- ❌ OS execution (Layer 5) not implemented

### For Developers

**Before modifying:**

1. Read [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
2. Run all tests: `pytest tests/ -v`
3. Follow existing code patterns in `src/semantic_hgr/`

**After changes:**

1. Run tests: `pytest tests/ -v`
2. Check formatting: `black src/ tests/`
3. Update documentation if architecture changes

---

## License & Citation

**Academic Use Only** - See LICENSE file for details.

If you use this work in research:

```bibtex
@misc{semantic-hgr-2026,
  author = {Neil Chang},
  title = {Semantic Hand Gesture Recognition using CLIP},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/neilharia7/Semantic-HGR-System}
}
```

**Research References:**

- **CLIP:** Radford et al. (2021) - Learning Transferable Visual Models From Natural Language Supervision
- **MediaPipe Hands:** Zhang et al. (2020) - MediaPipe Hands: On-device Real-time Hand Tracking
- **Temperature Scaling:** Guo et al. (2017) - On Calibration of Modern Neural Networks

---

## Contact & Support

**Project Maintainer:** Neil Chang  
**Institution:** Colorado State University (CSU)  
**Project Type:** Capstone/Thesis Research

**Documentation:**

- Architecture: [docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
- Installation: [docs/installation/INSTALL.md](docs/installation/INSTALL.md)
- Operations: [docs/operations/PRODUCTION_FINALIZATION_GUIDE.md](docs/operations/PRODUCTION_FINALIZATION_GUIDE.md)

**Known Limitations:**

- Static gestures only (no temporal dynamics)
- 19-intent vocabulary (not extensible without code changes)
- No command execution (demonstration only)
- CPU inference (slow without GPU)

---

## Acknowledgments

This project builds on:

- OpenCLIP (LAION-AI) - CLIP implementation
- MediaPipe (Google) - Hand detection
- PyTorch (Meta) - Deep learning framework
- HaGRID Dataset (Sber AI) - Gesture recognition research

**Supervisor:** [Add advisor name]  
**Academic Year:** 2025-2026

---

**Last Updated:** February 12, 2026  
**Repository Status:** Phase 1-2 Complete (Layers 1-3 operational)
