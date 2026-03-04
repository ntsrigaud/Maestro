# Repository Cleanup & Deletion Strategy — Pre-Architecture A

**Date**: March 4, 2026  
**Auditor Role**: Senior Software Architect / ML Infrastructure Auditor  
**Status**: Approved for Execution  
**Depends on**: [ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md](ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md)  
**Scope**: Controlled purge of CLIP-era artifacts; prepare experimental baseline for Dual-Stream Supervised Model

---

## SECTION 1 — FULL REPOSITORY INVENTORY

### 1.1 Core ML Modules (`src/semantic_hgr/`)

| File                                                                  | Purpose                                                                                                                                                                             | Era      | Flags                                                            |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------- |
| `src/semantic_hgr/__init__.py`                                        | Package root init                                                                                                                                                                   | CLIP-era | Exports CLIP-tier modules                                        |
| `src/semantic_hgr/representation/__init__.py`                         | Sub-package init for representation layer                                                                                                                                           | CLIP-era | References CLIPEncoder                                           |
| `src/semantic_hgr/representation/clip_encoder.py`                     | CLIP ViT-B/32 visual encoder; outputs 512-dim L2-normalized embeddings via `open_clip`; sole purpose is CLIP zero-shot representation                                               | CLIP-era | **Imports `open_clip`; entire module is CLIP**                   |
| `src/semantic_hgr/semantics/__init__.py`                              | Sub-package init for semantics layer                                                                                                                                                | CLIP-era | References SemanticIntentExtractor                               |
| `src/semantic_hgr/semantics/clip_intent.py`                           | CLIP-based zero-shot intent classifier; text prompt templates per class; cosine similarity head; `logit_scale` temperature; `SEMANTIC_INTENTS` dict; `STATIC_SEMANTIC_INTENTS` dict | CLIP-era | **764 lines of pure CLIP logic**                                 |
| `src/semantic_hgr/recognition_engine/__init__.py`                     | Recognition engine sub-package init                                                                                                                                                 | Mixed    | Needs review for CLIP exports                                    |
| `src/semantic_hgr/recognition_engine/hand_detection/__init__.py`      | Sub-package init for hand detection                                                                                                                                                 | Current  | No CLIP dependency                                               |
| `src/semantic_hgr/recognition_engine/hand_detection/hand_detector.py` | MediaPipe Hands wrapper; full 21-keypoint extraction with `HandDetectionResult`; bounding box computation                                                                           | Current  | **Required for Arch A**                                          |
| `src/semantic_hgr/recognition_engine/hand_detection/hand_data.py`     | `HandDetectionResult` dataclass; `landmarks: np.ndarray` shape `(21,3)` already defined; bbox, handedness, confidence, timestamp                                                    | Current  | **Required for Arch A; `landmarks` field is the keypoint input** |
| `src/semantic_hgr/recognition_engine/hand_detection/hand_cropper.py`  | Image cropper from bbox; comments reference `CLIPEncoder` and `open_clip` normalization convention                                                                                  | Mixed    | **REWRITE: logic needed, CLIP comments must be purged**          |
| `src/semantic_hgr/recognition_engine/preprocessing.py`                | `center_crop_frame()` center-crop + resize to 224×224; BGR→RGB; no CLIP import                                                                                                      | Current  | **Keep as-is**                                                   |
| `src/semantic_hgr/recognition_engine/temporal/__init__.py`            | Empty temporal module stub; no implementation                                                                                                                                       | Unknown  | **Orphaned stub — DELETE**                                       |
| `src/semantic_hgr/input_manager/__init__.py`                          | Input manager sub-package init                                                                                                                                                      | Current  | Camera pipeline, no CLIP                                         |
| `src/semantic_hgr/input_manager/input_manager.py`                     | Camera capture abstraction; frame buffering; CLIP-independent                                                                                                                       | Current  | Defer — not needed for Arch A experiments                        |
| `src/semantic_hgr/input_manager/example_usage.py`                     | Example script for input_manager                                                                                                                                                    | Unknown  | **Orphaned example — DELETE**                                    |
| `src/semantic_hgr/shared/__init__.py`                                 | Shared utilities init                                                                                                                                                               | Current  | No CLIP dependency                                               |
| `src/semantic_hgr/shared/logger.py`                                   | Structured logging utility                                                                                                                                                          | Current  | **Keep**                                                         |

### 1.2 Source Scripts (`src/semantic_hgr/scripts/`)

| File                                                              | Purpose                                                                                                                    | Era      | Flags                                                             |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------- |
| `src/semantic_hgr/scripts/__init__.py`                            | Scripts sub-package init                                                                                                   | CLIP-era | Registers CLIP console entry points                               |
| `src/semantic_hgr/scripts/static_eval.py`                         | Top-K evaluation framework; loads CLIPEncoder + SemanticIntentExtractor; computes Acc@1/3/5, confusion matrix, calibration | CLIP-era | **REWRITE: framework structure reusable; all internals are CLIP** |
| `src/semantic_hgr/scripts/realtime_demo.py`                       | Real-time CLIP gesture demo; webcam + CLIPEncoder + SemanticIntentExtractor pipeline                                       | CLIP-era | **DELETE: entirely CLIP**                                         |
| `src/semantic_hgr/scripts/validate_clip_integration.py`           | Validates CLIP embedding pipeline; 5 test classes each instantiating CLIPEncoder with ViT-B-32                             | CLIP-era | **DELETE: purpose is CLIP validation**                            |
| `src/semantic_hgr/scripts/validate_hand_detection.py`             | Validates MediaPipe hand detection on images; no CLIP dependency                                                           | Current  | **Keep**                                                          |
| `src/semantic_hgr/scripts/validate_input_layer.py`                | Validates camera/input pipeline; no CLIP dependency                                                                        | Current  | **Keep**                                                          |
| `src/semantic_hgr/scripts/archive/calibration_study.py`           | ECE/MCE/Brier metrics via CLIP pipeline; generated `results/calibration/calibration_report.json` (ECE=0.857)               | CLIP-era | **DELETE (already archived)**                                     |
| `src/semantic_hgr/scripts/archive/intent_overlap_analysis.py`     | CLIP embedding cosine similarity analysis between intent classes                                                           | CLIP-era | **DELETE (already archived)**                                     |
| `src/semantic_hgr/scripts/archive/prepare_calibration_dataset.py` | Prepares calibration dataset for CLIP pipeline                                                                             | CLIP-era | **DELETE (already archived)**                                     |

### 1.3 Top-Level Scripts (`scripts/`)

| File                                        | Purpose                                                     | Era      | Flags                                 |
| ------------------------------------------- | ----------------------------------------------------------- | -------- | ------------------------------------- |
| `scripts/diagnose_clip_confidence.py`       | Diagnoses CLIP softmax collapse and confidence distribution | CLIP-era | **DELETE**                            |
| `scripts/validate_clip_improvements.py`     | Validates CLIP temperature/prompt improvements              | CLIP-era | **DELETE**                            |
| `scripts/demo_hand_detection.py`            | MediaPipe hand detection demo; no CLIP                      | Current  | **Keep**                              |
| `scripts/validate_hand_detection_visual.py` | Visual validation of hand detection bbox                    | Current  | **Keep**                              |
| `scripts/download_hagrid_subset.py`         | Downloads HaGRID dataset subset via HuggingFace datasets    | Current  | **Keep**                              |
| `scripts/verify_install.py`                 | Install verification including CLIP model load test         | Mixed    | **REWRITE: remove CLIP verification** |

### 1.4 Archived Scripts (`scripts/archive/`)

| Group                             | Contents                                                                                                                                                                                                                                                                                          | Era    | Disposition                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------- |
| `scripts/archive/classification/` | 7 scripts: `build_no_gesture2_class.py`, `build_no_gesture2_simple.py`, `demo_gesture_recognition.py`, `real_time_inference.py`, `test_reproducibility.py`, `upload_model_to_huggingface.py`, `verify_clean_baseline.py`                                                                          | Legacy | **DELETE all**                                                          |
| `scripts/archive/dataset-utils/`  | 9 scripts: `audit_dataset_balance.py`, `capture_hand_samples.py`, `create_train_val_test_splits.py`, `download_celeba*.py`, `download_coco*.py`, `download_hagrid_kaggle.py`, `download_hagrid_subset.py`, `download_no_gesture2*.py`, `download_semantic_model.py`, `organize_hagrid_dataset.py` | Legacy | **DELETE all** (current `scripts/download_hagrid_subset.py` supersedes) |
| `scripts/archive/legacy-demos/`   | 12 scripts: shell scripts, event-based startup tests, threading tests, gesture demo, verify_environment, verify_install_old                                                                                                                                                                       | Legacy | **DELETE all**                                                          |
| `scripts/archive/environment.yml` | Old conda environment config                                                                                                                                                                                                                                                                      | Legacy | **DELETE**                                                              |

### 1.5 Notebooks (`notebooks/`)

| File                                              | Purpose                                                                                            | Era      | Flags                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------- |
| `notebooks/archive/finetune_vit_hagrid.ipynb`     | ViT fine-tuning on HaGRID — the supervised baseline that produced `class_metrics.json` (91–96% F1) | Current  | **KEEP — reference implementation for Architecture A Phase 2 (RGB stream)** |
| `notebooks/archive/lstm_train_vit_features.ipynb` | LSTM on ViT frame features — dynamic gesture track                                                 | Legacy   | **DELETE: dynamic gesture, no relevance to Architecture A**                 |
| `notebooks/archive/gesture_semantics.py`          | Exploratory CLIP-based gesture semantics script                                                    | CLIP-era | **DELETE**                                                                  |
| `notebooks/archive/inference_gesture.py`          | CLIP inference utility for gesture classification                                                  | CLIP-era | **DELETE**                                                                  |
| `notebooks/training_results/`                     | `class_metrics.json`, `confusion_matrix_absolute.csv`, `all_labels.npy`, `all_preds.npy`, etc.     | Current  | **Keep — baseline metrics for Table 1**                                     |

### 1.6 Tests (`tests/`)

| File                                   | Purpose                                                                                                     | Era      | Flags                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------- |
| `tests/test_clip_encoder.py`           | 278-line test suite for `CLIPEncoder`; validates CLIP ViT-B/32 embedding dims, determinism, normalization   | CLIP-era | **DELETE: tests a component being deleted** |
| `tests/test_semantic_intent.py`        | 315-line test suite for `SemanticIntentExtractor`; validates cosine similarity, text prompt encoding, top-K | CLIP-era | **DELETE: tests a component being deleted** |
| `tests/test_probability_validation.py` | 405-line test suite for softmax probability correctness in `SemanticIntentExtractor`                        | CLIP-era | **DELETE: tests CLIP probability pipeline** |

> **Note**: After deletion, there are zero tests. A new test suite targeting `HandDetector`, `HandDetectionResult`, `hand_cropper`, and `preprocessing` must be created as part of Architecture A setup.

### 1.7 Config & Environment Files

| File              | Purpose                                                                                                         | Era      | Flags                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `environment.yml` | Conda env (`gesture`); Python 3.10; OpenCV, NumPy, Pillow, tqdm, MediaPipe, datasets                            | Mixed    | **REWRITE: does not include torch, timm, torch_geometric; remove datasets if HaGRID already downloaded** |
| `pyproject.toml`  | Package config; dependencies include `open-clip-torch>=2.20.0`; console scripts include CLIP-based entry points | CLIP-era | **REWRITE: remove open-clip-torch, remove CLIP console scripts, update description and keywords**        |
| `Makefile`        | Make targets including `clip-check`, deprecated calibration/overlap targets                                     | Mixed    | **REWRITE: remove all CLIP targets, add Arch A training targets**                                        |

### 1.8 Documentation

| Path                                                       | Purpose                                                                 | Disposition                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------- |
| `docs/research/SYSTEM_AUDIT_AND_REDESIGN.md`               | Root cause analysis + Architecture A/B designs                          | **Keep**                                 |
| `docs/research/ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md` | Full Arch A execution plan                                              | **Keep**                                 |
| `docs/research/CLEANUP_AND_DELETION_STRATEGY.md`           | This document                                                           | **Keep**                                 |
| `docs/analysis/CALIBRATION_ROOT_CAUSE_ANALYSIS.md`         | Documents ECE=0.857 failure                                             | **Keep — Table 1 baseline evidence**     |
| `docs/analysis/INTENT_CLASSIFICATION.md`                   | CLIP intent classification analysis                                     | **DELETE**                               |
| `docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md`     | Mathematical spec of CLIP zero-shot pipeline                            | **DELETE: architecture being abandoned** |
| `docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md`        | System architecture doc for CLIP-tier system                            | **DELETE: architecture being abandoned** |
| `docs/development/PROJECT_PROGRESS.md`                     | Development progress log                                                | Keep (historical record)                 |
| `docs/development/RESEARCH_PROPOSAL_IEEE.md`               | IEEE research proposal                                                  | Keep                                     |
| `docs/development/work-logs/`                              | Work logs                                                               | Keep                                     |
| `docs/archive/`                                            | Archived docs — academic, deprecated, meeting-notes, research, training | **DELETE entire directory**              |
| `docs/legacy/`                                             | Legacy migration docs, naming conventions, cleanup summaries            | **DELETE entire directory**              |
| `docs/operations/PACKAGING_REFACTOR_REPORT.md`             | Packaging report                                                        | **DELETE**                               |
| `docs/operations/PRODUCTION_FINALIZATION_GUIDE.md`         | Production finalization for CLIP system                                 | **DELETE**                               |
| `docs/installation/INSTALL.md.old`                         | Old install guide                                                       | **DELETE**                               |
| `docs/installation/INSTALL.md`                             | Current install guide (references CLIP setup)                           | **REWRITE**                              |
| `notebooks/archive/GESTURE_SEMANTICS_ENHANCEMENT.md`       | CLIP-era enhancement guide                                              | **DELETE**                               |
| `notebooks/archive/NO_GESTURE2_INTEGRATION.md`             | Legacy no_gesture2 class integration                                    | **DELETE**                               |
| `notebooks/archive/QUICK_START_SEMANTIC.md`                | Quick start for CLIP semantic system                                    | **DELETE**                               |
| `notebooks/archive/README.md`                              | Archive README                                                          | **DELETE**                               |
| `notebooks/archive/SEMANTIC_GESTURE_RESEARCH.md`           | CLIP semantics research notes                                           | **DELETE**                               |
| `notebooks/archive/SEMANTIC_IMPROVEMENT_GUIDE.md`          | CLIP improvement guide                                                  | **DELETE**                               |
| `notebooks/archive/TRAINING_COMPARISON.md`                 | Training comparison (may contain ViT supervised data)                   | **DELETE**                               |
| `notebooks/DATASET_QUICKSTART.md`                          | HaGRID dataset quick start                                              | Keep                                     |

### 1.9 Data Directories

| Path                                                  | Contents                                               | Disposition |
| ----------------------------------------------------- | ------------------------------------------------------ | ----------- |
| `data/hagrid_subset/`                                 | 8 gesture class image directories                      | **Keep**    |
| `data/calibration/validation_gestures_synthetic.json` | Synthetic calibration dataset used for ECE=0.857 study | **DELETE**  |
| `results/calibration/calibration_report.json`         | ECE/MCE results (keep as evidence of baseline failure) | **Keep**    |
| `results/static_eval/`                                | Static evaluation results                              | Keep        |
| `results/overlap/`                                    | Intent overlap analysis results from CLIP study        | **DELETE**  |

---

## SECTION 2 — ARCHITECTURAL MISALIGNMENT DETECTION

Architecture A constraints: **fully supervised, GCN + ViT-B/16, no CLIP, no text prompts, no cosine similarity classification, no temperature scaling tied to logit_scale, no zero-shot evaluation**.

### 2.1 Classification Table

#### DELETE — No Longer Relevant

| File                                                              | Specific Violation                                                                                                                                                                                         | Justification                                                                                                                                                                                                |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/semantic_hgr/representation/clip_encoder.py`                 | Loads CLIP ViT-B/32 via `open_clip.create_model_and_transforms()`; outputs 512-dim CLIP embeddings                                                                                                         | Entire module is the component being replaced. No logic is salvageable — the new RGB encoder is `timm.create_model('vit_base_patch16_224')`, a fundamentally different initialization and inference contract |
| `src/semantic_hgr/semantics/clip_intent.py`                       | Text prompt generation (`STATIC_SEMANTIC_INTENTS` dict with 5 variants per class); `SemanticIntentExtractor` using `encode_text()` + cosine_similarity; `logit_scale` temperature; top-K via cosine scores | This is the classification head being entirely replaced by an MLP over fused embeddings                                                                                                                      |
| `src/semantic_hgr/scripts/realtime_demo.py`                       | Instantiates `CLIPEncoder` + `SemanticIntentExtractor`; full CLIP inference loop                                                                                                                           | Arch A will have its own inference script                                                                                                                                                                    |
| `src/semantic_hgr/scripts/validate_clip_integration.py`           | 5 test classes each constructing `CLIPEncoder(model_name='ViT-B-32')`; validates CLIP embedding geometry                                                                                                   | Validates a component that does not exist in Arch A                                                                                                                                                          |
| `src/semantic_hgr/scripts/archive/calibration_study.py`           | Runs ECE/MCE study on CLIP pipeline                                                                                                                                                                        | Study complete; results preserved in `calibration_report.json`                                                                                                                                               |
| `src/semantic_hgr/scripts/archive/intent_overlap_analysis.py`     | Computes cosine similarity between CLIP text embeddings                                                                                                                                                    | No role in supervised architecture                                                                                                                                                                           |
| `src/semantic_hgr/scripts/archive/prepare_calibration_dataset.py` | Creates dataset for CLIP calibration pipeline                                                                                                                                                              | Dataset no longer needed                                                                                                                                                                                     |
| `tests/test_clip_encoder.py`                                      | Tests `CLIPEncoder.encode_image()` output shape, normalization, determinism                                                                                                                                | Tests a deleted component                                                                                                                                                                                    |
| `tests/test_semantic_intent.py`                                   | Tests `SemanticIntentExtractor.extract_intent()` cosine similarity ranking                                                                                                                                 | Tests a deleted component                                                                                                                                                                                    |
| `tests/test_probability_validation.py`                            | Tests softmax probability output of `SemanticIntentExtractor`                                                                                                                                              | Tests CLIP softmax pipeline                                                                                                                                                                                  |
| `scripts/diagnose_clip_confidence.py`                             | Diagnoses CLIP confidence collapse via softmax histogram                                                                                                                                                   | Diagnostic for a deleted component                                                                                                                                                                           |
| `scripts/validate_clip_improvements.py`                           | Validates whether CLIP temperature/prompt changes improve accuracy                                                                                                                                         | Validates a deleted component                                                                                                                                                                                |
| `notebooks/archive/gesture_semantics.py`                          | CLIP-based gesture embedding exploration                                                                                                                                                                   | Replaced by GCN-based geometric embedding                                                                                                                                                                    |
| `notebooks/archive/inference_gesture.py`                          | CLIP gesture inference utility                                                                                                                                                                             | Replaced by Arch A inference                                                                                                                                                                                 |
| `notebooks/archive/lstm_train_vit_features.ipynb`                 | Dynamic gesture LSTM — entirely different track                                                                                                                                                            | Not part of static gesture Architecture A                                                                                                                                                                    |
| `data/calibration/validation_gestures_synthetic.json`             | Synthetic calibration data for CLIP study                                                                                                                                                                  | Study complete                                                                                                                                                                                               |
| `results/overlap/`                                                | CLIP text embedding overlap results                                                                                                                                                                        | No role in supervised architecture                                                                                                                                                                           |
| `docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md`            | Mathematical spec of CLIP zero-shot inference                                                                                                                                                              | Documents retired architecture                                                                                                                                                                               |
| `docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md`               | System design for CLIP-tier                                                                                                                                                                                | Documents retired architecture                                                                                                                                                                               |
| `docs/analysis/INTENT_CLASSIFICATION.md`                          | CLIP intent classification analysis                                                                                                                                                                        | Replaced by Architecture A gap analysis                                                                                                                                                                      |
| `scripts/archive/` (entire directory)                             | All 28 scripts: legacy demos, old dataset utils, old classification scripts                                                                                                                                | Pre-Architecture-A experimental detritus                                                                                                                                                                     |
| `docs/archive/` (entire directory)                                | Academic notes, deprecated docs, meeting notes for CLIP-era                                                                                                                                                | Historical noise                                                                                                                                                                                             |
| `docs/legacy/` (entire directory)                                 | Migration checklists, naming conventions for CLIP-era refactor                                                                                                                                             | Pre-Architecture-A administrative debt                                                                                                                                                                       |
| `docs/operations/PACKAGING_REFACTOR_REPORT.md`                    | Packaging report for CLIP package structure                                                                                                                                                                | Obsolete                                                                                                                                                                                                     |
| `docs/operations/PRODUCTION_FINALIZATION_GUIDE.md`                | Production guide for CLIP system                                                                                                                                                                           | Documents retired system                                                                                                                                                                                     |
| `docs/installation/INSTALL.md.old`                                | Superseded install guide                                                                                                                                                                                   | Superseded                                                                                                                                                                                                   |
| `notebooks/archive/GESTURE_SEMANTICS_ENHANCEMENT.md`              | CLIP semantic enhancement guide                                                                                                                                                                            | Documents retired research direction                                                                                                                                                                         |
| `notebooks/archive/NO_GESTURE2_INTEGRATION.md`                    | Legacy dataset integration                                                                                                                                                                                 | Obsolete dataset class                                                                                                                                                                                       |
| `notebooks/archive/QUICK_START_SEMANTIC.md`                       | CLIP quick-start                                                                                                                                                                                           | Retired                                                                                                                                                                                                      |
| `notebooks/archive/README.md`                                     | Archive README                                                                                                                                                                                             | Covered by main README                                                                                                                                                                                       |
| `notebooks/archive/SEMANTIC_GESTURE_RESEARCH.md`                  | CLIP research notes                                                                                                                                                                                        | Retired                                                                                                                                                                                                      |
| `notebooks/archive/SEMANTIC_IMPROVEMENT_GUIDE.md`                 | CLIP improvement guide                                                                                                                                                                                     | Retired                                                                                                                                                                                                      |
| `notebooks/archive/TRAINING_COMPARISON.md`                        | CLIP vs supervised training comparison                                                                                                                                                                     | Covered by SYSTEM_AUDIT_AND_REDESIGN.md                                                                                                                                                                      |
| `src/semantic_hgr/recognition_engine/temporal/__init__.py`        | Empty stub — no implementation, no imports                                                                                                                                                                 | Orphaned                                                                                                                                                                                                     |
| `src/semantic_hgr/input_manager/example_usage.py`                 | Example usage for input_manager                                                                                                                                                                            | Not part of Arch A experiment pipeline                                                                                                                                                                       |

#### REWRITE — Useful Structure, Incorrect Internals

| File                                                  | What Is Reusable                                                                                      | What Must Change                                                                                                                                                                                                                                | Priority                               |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `src/semantic_hgr/recognition_engine/hand_cropper.py` | Bounding-box-to-crop logic; padding; resizing                                                         | Remove all comments referencing `CLIPEncoder`, `open_clip`, CLIP-specific normalization; update docstring to reference ImageNet normalization for ViT-B/16                                                                                      | High (needed for RGB stream input)     |
| `src/semantic_hgr/scripts/static_eval.py`             | Dataset loading by class directory structure; top-K accuracy computation; confusion matrix generation | Replace `CLIPEncoder` + `SemanticIntentExtractor` imports and inference logic with Arch A `DualStreamModel` interface; retain metric computation                                                                                                | High (evaluation framework for Arch A) |
| `pyproject.toml`                                      | Build system config, package name, Python version, dev/test deps                                      | Remove `open-clip-torch`; update description + keywords; remove CLIP console scripts (`prepare-calibration-dataset`, `calibration-study`, `intent-overlap-analysis`, `realtime-demo`, `validate-clip-integration`); add new Arch A entry points | High                                   |
| `environment.yml`                                     | Conda channel config, Python 3.10, OpenCV, NumPy, Pillow, tqdm, MediaPipe                             | Add `torch`, `torchvision`, `timm`, `torch_geometric`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`; remove CLIP-era pip packages                                                                                                          | High                                   |
| `Makefile`                                            | Build structure, install targets, clean targets                                                       | Remove `clip-check` target; remove deprecated `calibration` and `overlap-analysis` comments; add `train-gcn`, `train-rgb`, `train-dual`, `validate-keypoints`, `eval` targets                                                                   | Medium                                 |
| `scripts/verify_install.py`                           | Basic import verification structure                                                                   | Remove CLIP model load test; add `timm`, `torch_geometric`, `mediapipe` verification                                                                                                                                                            | Medium                                 |
| `docs/installation/INSTALL.md`                        | Setup instructions structure                                                                          | Remove CLIP model download instructions; add `timm` and `torch_geometric` install steps                                                                                                                                                         | Low                                    |

#### KEEP — Compatible With Architecture A

| File                                                                  | Reason                                                                                                                                    |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `src/semantic_hgr/recognition_engine/hand_detection/hand_detector.py` | MediaPipe Hands wrapper; extracts 21 keypoints stored in `HandDetectionResult.landmarks (21,3)`; directly provides geometric stream input |
| `src/semantic_hgr/recognition_engine/hand_detection/hand_data.py`     | `HandDetectionResult` dataclass with `landmarks: np.ndarray` shape `(21,3)` is the exact contract for the GCN input pipeline              |
| `src/semantic_hgr/recognition_engine/preprocessing.py`                | `center_crop_frame()` is the RGB stream preprocessing function; no CLIP dependency                                                        |
| `src/semantic_hgr/shared/logger.py`                                   | Logging utility; no ML dependencies                                                                                                       |
| `src/semantic_hgr/scripts/validate_hand_detection.py`                 | Validates MediaPipe keypoint extraction; directly implements Section 2.6 pre-training validation checklist gate                           |
| `src/semantic_hgr/scripts/validate_input_layer.py`                    | Camera pipeline validation; reusable for Arch A inference                                                                                 |
| `scripts/demo_hand_detection.py`                                      | MediaPipe detection demo; useful for visual keypoint inspection                                                                           |
| `scripts/validate_hand_detection_visual.py`                           | Visual keypoint validation; required for Gate 1 verification                                                                              |
| `scripts/download_hagrid_subset.py`                                   | HaGRID data download; required for Arch A data pipeline                                                                                   |
| `notebooks/archive/finetune_vit_hagrid.ipynb`                         | Reference ViT fine-tuning implementation; direct precursor to Arch A Phase 2 RGB stream training                                          |
| `notebooks/training_results/*.json, *.csv, *.npy`                     | Supervised baseline metrics (F1 91–96%); required for Table 1 comparison                                                                  |
| `data/hagrid_subset/`                                                 | 8-class image data already downloaded                                                                                                     |
| `results/calibration/calibration_report.json`                         | ECE=0.857 baseline evidence for Table 1                                                                                                   |
| `results/static_eval/`                                                | Existing evaluation results                                                                                                               |
| `docs/research/SYSTEM_AUDIT_AND_REDESIGN.md`                          | Research foundation document                                                                                                              |
| `docs/research/ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md`            | Execution blueprint                                                                                                                       |
| `docs/analysis/CALIBRATION_ROOT_CAUSE_ANALYSIS.md`                    | Root cause documentation (Table 1 evidence)                                                                                               |
| `docs/development/PROJECT_PROGRESS.md`                                | Progress tracking                                                                                                                         |
| `docs/development/RESEARCH_PROPOSAL_IEEE.md`                          | Conference target documentation                                                                                                           |
| `notebooks/DATASET_QUICKSTART.md`                                     | HaGRID dataset reference                                                                                                                  |

---

## SECTION 3 — CLEAN STRUCTURE TARGET BLUEPRINT

The following is the canonical post-cleanup repository structure for Architecture A:

```
Semantic-HGR-System/
│
├── data/
│   ├── hagrid_subset/       ← existing 8-class image data (keep)
│   │   ├── thumbs_up/
│   │   ├── thumbs_down/
│   │   ├── open_palm/
│   │   ├── closed_fist/
│   │   ├── peace_sign/
│   │   ├── ok_sign/
│   │   ├── pointing/
│   │   └── call_gesture/
│   ├── splits/              ← new: deterministic 70/15/15 split json (created by extract_keypoints.py)
│   └── keypoints/           ← new: NPZ caches (train/val/test keypoints + failed_detection.txt)
│
├── src/
│   └── semantic_hgr/
│       ├── __init__.py
│       ├── shared/
│       │   ├── __init__.py
│       │   └── logger.py
│       ├── recognition_engine/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   └── hand_detection/
│       │       ├── __init__.py
│       │       ├── hand_data.py
│       │       ├── hand_detector.py
│       │       └── hand_cropper.py        ← rewritten (CLIP comments purged)
│       ├── models/                        ← new directory for Arch A models
│       │   ├── __init__.py
│       │   ├── gcn.py                     ← GCN stream (Section 3 of blueprint)
│       │   ├── vit_rgb.py                 ← ViT-B/16 RGB stream (Section 4)
│       │   ├── fusion.py                  ← Cross-attention fusion (Section 5)
│       │   └── dual_stream.py             ← Assembled dual-stream model
│       ├── training/                      ← new directory for training loops
│       │   ├── __init__.py
│       │   ├── train_gcn.py               ← Phase 1 standalone GCN training
│       │   ├── train_rgb.py               ← Phase 2 ViT-B/16 baseline training
│       │   └── train_dual.py              ← Phase 3 + 4 joint training
│       └── evaluation/                    ← rewritten from static_eval.py
│           ├── __init__.py
│           ├── metrics.py                 ← Acc@1/3/5, F1, ECE, Brier
│           ├── calibration.py             ← Reliability diagram, ECE computation
│           ├── confusion_matrix.py        ← Confusion matrix + heatmap
│           └── statistical_tests.py       ← McNemar's test, bootstrap CI
│
├── scripts/
│   ├── extract_keypoints.py               ← new: offline MediaPipe keypoint extraction
│   ├── validate_keypoints.py              ← new: pre-training validation checklist (Section 2.6)
│   ├── download_hagrid_subset.py          ← keep
│   ├── demo_hand_detection.py             ← keep
│   └── validate_hand_detection_visual.py  ← keep
│
├── configs/
│   ├── arch_a_gcn.yaml                    ← new: GCN hyperparameters
│   ├── arch_a_rgb.yaml                    ← new: ViT-B/16 fine-tuning schedule
│   └── arch_a_dual.yaml                   ← new: joint training config
│
├── notebooks/
│   ├── DATASET_QUICKSTART.md              ← keep
│   ├── reference/                         ← renamed from archive/
│   │   ├── finetune_vit_hagrid.ipynb      ← keep (supervised baseline reference)
│   │   └── training_results/              ← keep (baseline metrics)
│   └── experiments/                       ← new: Arch A experiment notebooks
│
├── results/
│   ├── calibration/                       ← keep (baseline evidence)
│   ├── static_eval/                       ← keep
│   ├── keypoint_validation/               ← new: t-SNE visualizations from Section 2.6
│   └── geometry_stream/                   ← new: GCN diagnostic outputs
│
├── tests/
│   ├── README.md
│   ├── test_hand_detector.py              ← new: replaces deleted CLIP tests
│   ├── test_hand_data.py                  ← new: validates landmark extraction contract
│   ├── test_preprocessing.py              ← new: validates center_crop_frame
│   └── test_gcn.py                        ← new: validates GCN graph construction
│
├── docs/
│   ├── research/
│   │   ├── SYSTEM_AUDIT_AND_REDESIGN.md
│   │   ├── ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md
│   │   └── CLEANUP_AND_DELETION_STRATEGY.md
│   ├── analysis/
│   │   └── CALIBRATION_ROOT_CAUSE_ANALYSIS.md
│   ├── development/
│   │   ├── PROJECT_PROGRESS.md
│   │   ├── RESEARCH_PROPOSAL_IEEE.md
│   │   └── work-logs/
│   └── installation/
│       └── INSTALL.md
│
├── environment.yml                        ← rewritten for Arch A
├── pyproject.toml                         ← rewritten, CLIP deps removed
├── Makefile                               ← rewritten, Arch A targets
└── README.md
```

---

## SECTION 4 — DELETION PLAN

### Stage 1 — Remove Legacy CLIP Model Backbones

**Target: Core CLIP ML modules**

| File to Delete                                    | Risk      | Dependency Impact                                                                                                                                                                            |
| ------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/semantic_hgr/representation/clip_encoder.py` | 🟡 Medium | Imported by `static_eval.py`, `realtime_demo.py`, `validate_clip_integration.py`, `archive/calibration_study.py`, `archive/intent_overlap_analysis.py` — all of which are also being deleted |
| `src/semantic_hgr/representation/__init__.py`     | 🟢 Low    | Only exports CLIPEncoder; sub-package being repurposed                                                                                                                                       |
| `src/semantic_hgr/semantics/clip_intent.py`       | 🟡 Medium | Imported by all scripts in Stage 1 above; all importers deleted in this same stage                                                                                                           |
| `src/semantic_hgr/semantics/__init__.py`          | 🟢 Low    | Only re-exports CLIP intent classes                                                                                                                                                          |

**Verification after Stage 1**: `grep -r "clip_encoder\|clip_intent\|CLIPEncoder\|SemanticIntentExtractor" src/ tests/ scripts/ --include="*.py"` must return zero hits from retained files.

---

### Stage 2 — Remove CLIP-Dependent Scripts and Tests

**Target: All scripts and tests whose sole purpose is CLIP validation/inference**

| File/Directory to Delete                                          | Risk   | Notes                                                 |
| ----------------------------------------------------------------- | ------ | ----------------------------------------------------- |
| `src/semantic_hgr/scripts/realtime_demo.py`                       | 🟢 Low | Standalone; no retained module imports it             |
| `src/semantic_hgr/scripts/validate_clip_integration.py`           | 🟢 Low | Standalone validation script                          |
| `src/semantic_hgr/scripts/archive/calibration_study.py`           | 🟢 Low | Archive; results already in `calibration_report.json` |
| `src/semantic_hgr/scripts/archive/intent_overlap_analysis.py`     | 🟢 Low | Archive                                               |
| `src/semantic_hgr/scripts/archive/prepare_calibration_dataset.py` | 🟢 Low | Archive                                               |
| `tests/test_clip_encoder.py`                                      | 🟢 Low | Tests deleted component                               |
| `tests/test_semantic_intent.py`                                   | 🟢 Low | Tests deleted component                               |
| `tests/test_probability_validation.py`                            | 🟢 Low | Tests deleted component                               |
| `scripts/diagnose_clip_confidence.py`                             | 🟢 Low | Imports CLIPEncoder                                   |
| `scripts/validate_clip_improvements.py`                           | 🟢 Low | Imports CLIPEncoder + SemanticIntentExtractor         |
| `notebooks/archive/gesture_semantics.py`                          | 🟢 Low | Standalone CLIP exploration                           |
| `notebooks/archive/inference_gesture.py`                          | 🟢 Low | Standalone CLIP inference                             |

**Verification after Stage 2**: `python -c "from semantic_hgr.scripts.validate_hand_detection import main"` succeeds. `python -c "from semantic_hgr.scripts.validate_input_layer import main"` succeeds.

---

### Stage 3 — Remove Deprecated Experiments and Legacy Archives

**Target: Entire archive directories and stale experiment notebooks**

| Path to Delete                                             | Risk   | Notes                                                  |
| ---------------------------------------------------------- | ------ | ------------------------------------------------------ |
| `scripts/archive/` (entire directory — 28 files)           | 🟢 Low | No retained module references any file here            |
| `notebooks/archive/lstm_train_vit_features.ipynb`          | 🟢 Low | Dynamic gesture track; no relevance                    |
| `notebooks/archive/GESTURE_SEMANTICS_ENHANCEMENT.md`       | 🟢 Low | CLIP-era doc                                           |
| `notebooks/archive/NO_GESTURE2_INTEGRATION.md`             | 🟢 Low | Legacy dataset class                                   |
| `notebooks/archive/QUICK_START_SEMANTIC.md`                | 🟢 Low | CLIP quick-start                                       |
| `notebooks/archive/README.md`                              | 🟢 Low | Archive README                                         |
| `notebooks/archive/SEMANTIC_GESTURE_RESEARCH.md`           | 🟢 Low | CLIP research notes                                    |
| `notebooks/archive/SEMANTIC_IMPROVEMENT_GUIDE.md`          | 🟢 Low | CLIP improvement guide                                 |
| `notebooks/archive/TRAINING_COMPARISON.md`                 | 🟢 Low | Covered by audit doc                                   |
| `src/semantic_hgr/recognition_engine/temporal/__init__.py` | 🟢 Low | Empty stub                                             |
| `src/semantic_hgr/input_manager/example_usage.py`          | 🟢 Low | Orphaned example file                                  |
| `data/calibration/` (entire directory)                     | 🟢 Low | Synthetic dataset used only for CLIP calibration study |
| `results/overlap/` (entire directory)                      | 🟢 Low | CLIP intent overlap analysis outputs                   |

**Verification after Stage 3**: `find scripts/ notebooks/ -name "*.py" -o -name "*.ipynb" -o -name "*.sh" | xargs grep -l "clip\|CLIP\|open_clip" 2>/dev/null` returns empty.

---

### Stage 4 — Remove Obsolete Configuration

**Target: CLIP-specific package config entries and old environment spec**

| File              | Action                                                                                                                                                                                                                                 | Risk                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `pyproject.toml`  | Remove `open-clip-torch` from `dependencies`; remove console scripts: `prepare-calibration-dataset`, `calibration-study`, `intent-overlap-analysis`, `realtime-demo`, `validate-clip-integration`; update `description` and `keywords` | 🟡 Medium (package re-install required) |
| `environment.yml` | Add Architecture A dependencies; remove `datasets` install if HaGRID already downloaded                                                                                                                                                | 🟡 Medium (env rebuild required)        |
| `Makefile`        | Remove `clip-check`, `calibration-interactive`, `calibration-evaluate` targets; the `realtime-demo` target; remove deprecated comment lines about calibration/overlap                                                                  | 🟢 Low                                  |

**Verification after Stage 4**: `grep -n "clip\|CLIP\|open_clip\|calibration-study\|intent-overlap" pyproject.toml` returns zero hits.

---

### Stage 5 — Remove Redundant Documentation

| Path                                                   | Action | Risk   |
| ------------------------------------------------------ | ------ | ------ |
| `docs/archive/` (entire directory)                     | Delete | 🟢 Low |
| `docs/legacy/` (entire directory)                      | Delete | 🟢 Low |
| `docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md` | Delete | 🟢 Low |
| `docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md`    | Delete | 🟢 Low |
| `docs/analysis/INTENT_CLASSIFICATION.md`               | Delete | 🟢 Low |
| `docs/operations/PACKAGING_REFACTOR_REPORT.md`         | Delete | 🟢 Low |
| `docs/operations/PRODUCTION_FINALIZATION_GUIDE.md`     | Delete | 🟢 Low |
| `docs/installation/INSTALL.md.old`                     | Delete | 🟢 Low |

**Verification after Stage 5**: `find docs/ -name "*.md" | wc -l` returns ≤ 10 (the 10 kept files enumerated in Section 3). `grep -r "CLIP\|zero-shot\|logit_scale\|cosine_similarity" docs/ --include="*.md"` returns only hits inside `SYSTEM_AUDIT_AND_REDESIGN.md` and `CALIBRATION_ROOT_CAUSE_ANALYSIS.md` (which intentionally document the failure).

---

## SECTION 5 — SAFETY CHECKS BEFORE DELETION

### 5.1 Static Import Graph — Retained Module Dependencies

The following dependency graph covers **retained modules only**. No retained module depends on any file scheduled for deletion.

```
hand_detector.py
   └── imports: cv2, numpy, mediapipe, hand_data.py, shared/logger.py

hand_data.py
   └── imports: numpy, dataclasses (stdlib)

hand_cropper.py
   └── imports: cv2, numpy, PIL, hand_data.py  [no open_clip]

preprocessing.py
   └── imports: cv2, numpy (stdlib)

shared/logger.py
   └── imports: logging (stdlib)

scripts/validate_hand_detection.py
   └── imports: hand_detector.py, hand_data.py, cv2

scripts/validate_input_layer.py
   └── imports: input_manager.py, cv2

scripts/demo_hand_detection.py
   └── imports: hand_detector.py, cv2

scripts/validate_hand_detection_visual.py
   └── imports: hand_detector.py, cv2, matplotlib
```

**Nothing in the retained set imports `clip_encoder.py` or `clip_intent.py`.**

### 5.2 What Will Break If Deleted — and Why It Is Safe

| Deleted File                 | Imported By (to be confirmed)                                                                                                                                                            | Safe?                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `clip_encoder.py`            | `static_eval.py` (rewrite), `realtime_demo.py` (delete), `validate_clip_integration.py` (delete), `archive/calibration_study.py` (delete), `archive/intent_overlap_analysis.py` (delete) | ✅ All importers are also deleted or rewritten |
| `clip_intent.py`             | Same 5 files as above + `tests/test_semantic_intent.py` (delete), `tests/test_probability_validation.py` (delete)                                                                        | ✅ All importers are deleted                   |
| All test files               | No production module imports test files                                                                                                                                                  | ✅ Safe                                        |
| `scripts/archive/` all files | Not imported by any retained module (confirmed by grep: zero `import` hits into this directory from outside)                                                                             | ✅ Safe                                        |

### 5.3 Circular Import Detection

Current import graph has no circular imports within the retained set. The CLIP-era modules (`clip_encoder` → `open_clip`, `clip_intent` → `clip_encoder`) form a leaf dependency tree — no retained module sits in their import path.

Verify after deletion:

```bash
python -c "
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult
from semantic_hgr.recognition_engine.preprocessing import center_crop_frame
from semantic_hgr.shared.logger import get_logger
print('All retained imports OK')
"
```

### 5.4 Test Coverage Impact

**Before deletion**: 3 test files, ~998 lines, 100% targeting CLIP components being deleted.  
**After deletion**: 0 test files remaining (test suite is entirely CLIP-era).

This is acceptable because:

1. All three test files test deleted components — retaining them would cause `ImportError` after deletion
2. The `hand_detector.py`, `hand_data.py`, `hand_cropper.py`, and `preprocessing.py` retained modules are currently **untested** regardless
3. New tests must be written for Architecture A components (see Section 3 target structure)

**Action required**: Create `tests/test_hand_detector.py`, `tests/test_hand_data.py`, `tests/test_preprocessing.py` as the first post-cleanup task.

### 5.5 Environment Dependency Confirmation

**Package `open-clip-torch`**: Used exclusively by `clip_encoder.py`. No retained module imports `open_clip`. Safe to remove from `pyproject.toml` and `environment.yml`.

**Package `datasets`**: Used by `scripts/download_hagrid_subset.py` (retained) and `scripts/archive/dataset-utils/download_hagrid_subset.py` (deleted). The top-level script is retained, so `datasets` must remain as an optional dependency.

**Package `sentence-transformers`**: Listed as `optional-dependencies[semantic-matching]`. Not imported anywhere in the current active codebase (Architecture B future work). Keep as optional, do not add to base deps.

---

## SECTION 6 — ENVIRONMENT SIMPLIFICATION

### 6.1 Current vs Required Dependencies

| Package                       | Current Status                 | Architecture A Required                  | Action                            |
| ----------------------------- | ------------------------------ | ---------------------------------------- | --------------------------------- |
| `numpy>=1.24.0`               | Required                       | ✅ Yes                                   | Keep                              |
| `pillow>=9.0.0`               | Required                       | ✅ Yes (image loading)                   | Keep                              |
| `opencv-python>=4.8.0`        | Required                       | ✅ Yes (resize, crop, BGR→RGB)           | Keep                              |
| `mediapipe==0.10.21`          | Required                       | ✅ Yes (keypoint extraction)             | Keep; pin version                 |
| `torch>=2.0.0`                | Required                       | ✅ Yes                                   | Keep                              |
| `torchvision>=0.15.0`         | Required                       | ✅ Yes                                   | Keep                              |
| **`open-clip-torch>=2.20.0`** | Required                       | ❌ **No**                                | **REMOVE**                        |
| `matplotlib>=3.5.0`           | Required                       | ✅ Yes (evaluation plots)                | Keep                              |
| `seaborn>=0.11.0`             | Required                       | ✅ Yes (confusion matrix heatmap)        | Keep                              |
| `scikit-learn>=1.0.0`         | Required                       | ✅ Yes (metrics, calibration)            | Keep                              |
| `tqdm>=4.65.0`                | Required                       | ✅ Yes                                   | Keep                              |
| **`timm`**                    | **Not listed**                 | ✅ **Required** — `vit_base_patch16_224` | **ADD**                           |
| **`torch-geometric`**         | **Not listed**                 | ✅ **Required** — `GCNConv`              | **ADD**                           |
| `pandas`                      | Not listed                     | ✅ Recommended (result tables)           | ADD                               |
| `datasets>=2.0.0`             | Optional `[download]`          | ✅ Optional (HaGRID download)            | Keep as optional                  |
| `sentence-transformers`       | Optional `[semantic-matching]` | ❌ Architecture B only                   | Keep as optional; do not activate |

### 6.2 Architecture A Minimal `environment.yml`

```yaml
name: gesture
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10

  # Core computation
  - conda-forge::numpy>=1.24.0
  - conda-forge::pillow>=9.0.0
  - conda-forge::opencv>=4.8.0
  - conda-forge::tqdm>=4.65.0
  - conda-forge::pandas>=1.5.0

  # Deep learning — PyTorch
  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision>=0.15.0

  # Evaluation and visualization
  - conda-forge::scikit-learn>=1.0.0
  - conda-forge::matplotlib>=3.5.0
  - conda-forge::seaborn>=0.11.0

  - pip
  - pip:
      - mediapipe==0.10.21
      - timm>=0.9.0 # ViT-B/16 vit_base_patch16_224
      - torch-geometric>=2.3.0 # GCNConv
      - datasets>=2.0.0 # HaGRID download (optional after first use)
```

### 6.3 Architecture A Minimal `pyproject.toml` Dependencies Block

```toml
dependencies = [
    "numpy>=1.24.0,<2.0.0",
    "pillow>=9.0.0,<13.0.0",
    "opencv-python>=4.8.0",
    "mediapipe==0.10.21",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "timm>=0.9.0",
    "torch-geometric>=2.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.5.0",
    "tqdm>=4.65.0",
]
```

**Removed**: `open-clip-torch`  
**Added**: `timm`, `torch-geometric`, `pandas`

**Total dependency reduction**: 1 package removed (`open-clip-torch`, ~500MB model weights pulled on first use), 3 canonical Arch A packages added.

---

## SECTION 7 — POST-CLEANUP VALIDATION CHECKLIST

Execute these checks in order after all deletion stages are complete.

### 7.1 Environment Health

```bash
# Step 1: Rebuild environment from updated environment.yml
conda env create -f environment.yml --name gesture-arch-a
conda activate gesture-arch-a

# Step 2: Install package
pip install -e .

# Expected: No errors; no CLIP-related package download
```

### 7.2 Zero CLIP Reference Verification

```bash
# Must return ZERO hits from non-documentation source files
grep -r "open_clip\|CLIPEncoder\|SemanticIntentExtractor\|logit_scale\|cosine_similarity\|text_embed\|encode_text\|zero.shot" \
  src/ scripts/ tests/ notebooks/ configs/ \
  --include="*.py" --include="*.yaml" --include="*.yml" --include="*.toml" \
  2>/dev/null
```

Expected output: empty (no matches).

```bash
# Verify no CLIP imports remain
grep -r "from semantic_hgr.representation.clip_encoder\|from semantic_hgr.semantics" \
  src/ scripts/ tests/ --include="*.py" 2>/dev/null
```

Expected output: empty.

### 7.3 Retained Module Import Test

```bash
python -c "
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult
from semantic_hgr.recognition_engine.hand_detection.hand_cropper import HandCropper
from semantic_hgr.recognition_engine.preprocessing import center_crop_frame
from semantic_hgr.shared.logger import get_logger
print('Retained module imports: OK')
"
```

### 7.4 Architecture A Dependency Import Test

```bash
python -c "
import torch
import torchvision
import timm
import torch_geometric
import mediapipe
import cv2
import numpy
import sklearn
import matplotlib
import seaborn
import pandas

# Verify ViT-B/16 model loads
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=8)
print(f'ViT-B/16 created: {sum(p.numel() for p in model.parameters()):,} params')

# Verify GCNConv available
from torch_geometric.nn import GCNConv
conv = GCNConv(3, 64)
print(f'GCNConv created: {sum(p.numel() for p in conv.parameters())} params')

print('Architecture A dependency imports: OK')
"
```

### 7.5 Script Smoke Tests

```bash
# These must execute without ImportError
python scripts/verify_install.py
python scripts/demo_hand_detection.py --help 2>/dev/null || true
python -c "from src.semantic_hgr.scripts.validate_hand_detection import main; print('validate_hand_detection: OK')"
```

### 7.6 No Dead `__init__.py` Exports

```bash
# Verify __init__.py files do not export deleted symbols
grep -r "CLIPEncoder\|SemanticIntentExtractor\|clip_encoder\|clip_intent" \
  src/semantic_hgr/__init__.py \
  src/semantic_hgr/representation/__init__.py \
  src/semantic_hgr/semantics/__init__.py 2>/dev/null
```

Expected output: empty.

### 7.7 Repository Keyword Search (Final Gate)

```bash
# Final confirmation — search entire repo
grep -r "clip" . \
  --include="*.py" --include="*.toml" --include="*.yml" --include="*.yaml" \
  --exclude-dir=.git --exclude-dir=.venv \
  -i 2>/dev/null | grep -v "docs/research\|docs/analysis" | grep -v ".pyc"
```

Expected: Zero hits outside documentation files that intentionally reference CLIP as a historical baseline.

### 7.8 Data Integrity Check

```bash
# Verify image data is intact
for class in thumbs_up thumbs_down open_palm closed_fist peace_sign ok_sign pointing call_gesture; do
  count=$(find data/hagrid_subset/$class -name "*.jpg" | wc -l)
  echo "$class: $count images"
done
```

Expected: Each class shows the same count as before cleanup (image data is not touched).

---

## SECTION 8 — EXECUTION ORDER

```
STEP 1: GIT BASELINE
─────────────────────────────────────────────────────
[ ] git add -A
[ ] git commit -m "chore: pre-cleanup snapshot — CLIP-era codebase"
[ ] git checkout -b arch-a-cleanup
    (All cleanup work isolated to this branch)

STEP 2: STAGE 1 — Remove Legacy CLIP Model Backbones
─────────────────────────────────────────────────────
[ ] rm src/semantic_hgr/representation/clip_encoder.py
[ ] rm src/semantic_hgr/representation/__init__.py
[ ] rm src/semantic_hgr/semantics/clip_intent.py
[ ] rm src/semantic_hgr/semantics/__init__.py
[ ] Verification: grep -r "from semantic_hgr.representation.clip_encoder\|from semantic_hgr.semantics" src/ tests/ scripts/ --include="*.py" → empty

STEP 3: STAGE 2 — Remove CLIP-Dependent Scripts and Tests
─────────────────────────────────────────────────────────
[ ] rm src/semantic_hgr/scripts/realtime_demo.py
[ ] rm src/semantic_hgr/scripts/validate_clip_integration.py
[ ] rm src/semantic_hgr/scripts/archive/calibration_study.py
[ ] rm src/semantic_hgr/scripts/archive/intent_overlap_analysis.py
[ ] rm src/semantic_hgr/scripts/archive/prepare_calibration_dataset.py
[ ] rm tests/test_clip_encoder.py
[ ] rm tests/test_semantic_intent.py
[ ] rm tests/test_probability_validation.py
[ ] rm scripts/diagnose_clip_confidence.py
[ ] rm scripts/validate_clip_improvements.py
[ ] rm notebooks/archive/gesture_semantics.py
[ ] rm notebooks/archive/inference_gesture.py
[ ] Verification: python -m pytest tests/ → 0 tests, 0 errors (no failing imports)

STEP 4: STAGE 3 — Remove Deprecated Experiments and Archives
─────────────────────────────────────────────────────────────
[ ] rm -rf scripts/archive/
[ ] rm notebooks/archive/lstm_train_vit_features.ipynb
[ ] rm notebooks/archive/GESTURE_SEMANTICS_ENHANCEMENT.md
[ ] rm notebooks/archive/NO_GESTURE2_INTEGRATION.md
[ ] rm notebooks/archive/QUICK_START_SEMANTIC.md
[ ] rm notebooks/archive/README.md
[ ] rm notebooks/archive/SEMANTIC_GESTURE_RESEARCH.md
[ ] rm notebooks/archive/SEMANTIC_IMPROVEMENT_GUIDE.md
[ ] rm notebooks/archive/TRAINING_COMPARISON.md
[ ] rm src/semantic_hgr/recognition_engine/temporal/__init__.py
[ ] rm src/semantic_hgr/input_manager/example_usage.py
[ ] rm -rf data/calibration/
[ ] rm -rf results/overlap/
[ ] Verification: find . -empty -type d (remove empty dirs)

STEP 5: STAGE 4 — Rewrite Configuration Files
───────────────────────────────────────────────
[ ] Edit pyproject.toml:
    - Remove `open-clip-torch` from dependencies
    - Remove console script entries (5 CLIP-based scripts)
    - Update description: "Dual-Stream Gesture Recognition System for IEEE Research"
    - Update keywords: remove "clip", add "graph-neural-network", "dual-stream"
[ ] Edit environment.yml:
    - Add pytorch channel
    - Add timm, torch-geometric, pandas
    - Remove open-clip-torch (if present)
[ ] Edit Makefile:
    - Remove clip-check target
    - Remove deprecated calibration/overlap comment lines
    - Add train-gcn, train-rgb, train-dual, validate-keypoints, eval stub targets
[ ] Rewrite scripts/verify_install.py:
    - Remove CLIPEncoder import test
    - Add timm, torch_geometric, mediapipe
[ ] Verification: grep -n "clip\|open_clip" pyproject.toml environment.yml Makefile → empty

STEP 6: STAGE 5 — Remove Redundant Documentation
───────────────────────────────────────────────────
[ ] rm -rf docs/archive/
[ ] rm -rf docs/legacy/
[ ] rm docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md
[ ] rm docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md
[ ] rm docs/analysis/INTENT_CLASSIFICATION.md
[ ] rm docs/operations/PACKAGING_REFACTOR_REPORT.md
[ ] rm docs/operations/PRODUCTION_FINALIZATION_GUIDE.md
[ ] rm docs/installation/INSTALL.md.old
[ ] Verification: find docs/ -name "*.md" | sort (review remaining list)

STEP 7: EXECUTE REWRITES
─────────────────────────
[ ] Rewrite src/semantic_hgr/recognition_engine/hand_detection/hand_cropper.py
    (Remove CLIP-referencing comments; update normalization docstring to ImageNet stats)
[ ] Rewrite src/semantic_hgr/scripts/static_eval.py → move to src/semantic_hgr/evaluation/
    (Replace CLIPEncoder/SemanticIntentExtractor with DualStreamModel interface stub)
[ ] Update src/semantic_hgr/__init__.py
    (Remove exports of CLIPEncoder/SemanticIntentExtractor; add Arch A module paths)
[ ] Update docs/installation/INSTALL.md
    (Remove CLIP model download; add timm, torch-geometric install steps)

STEP 8: POST-CLEANUP VALIDATION
────────────────────────────────
[ ] Run Section 7.1 — Environment rebuild and install
[ ] Run Section 7.2 — Zero CLIP reference verification
[ ] Run Section 7.3 — Retained module import test
[ ] Run Section 7.4 — Architecture A dependency import test
[ ] Run Section 7.5 — Script smoke tests
[ ] Run Section 7.6 — __init__.py exports clean
[ ] Run Section 7.7 — Repository keyword search final gate
[ ] Run Section 7.8 — Data integrity check

STEP 9: CREATE ARCHITECTURE A STUBS
────────────────────────────────────
[ ] mkdir -p src/semantic_hgr/models/
[ ] mkdir -p src/semantic_hgr/training/
[ ] mkdir -p src/semantic_hgr/evaluation/
[ ] mkdir -p configs/
[ ] mkdir -p data/splits/ data/keypoints/
[ ] mkdir -p results/keypoint_validation/ results/geometry_stream/
[ ] Create placeholder __init__.py in each new directory
[ ] Create configs/arch_a_gcn.yaml, arch_a_rgb.yaml, arch_a_dual.yaml stubs

STEP 10: COMMIT CLEAN BASELINE
────────────────────────────────
[ ] git add -A
[ ] git commit -m "refactor: clean CLIP-era artifacts; Architecture A baseline structure"
[ ] (Optional) git tag v0.2.0-arch-a-baseline

FINAL STATE: Repository is CLIP-free and Architecture A ready
```

---

## ARCHITECTURE A READINESS CRITERIA

The repository is confirmed **ready for Architecture A implementation** when ALL of the following are true:

| Criterion                           | Verification Command                                                                                                                                                                                                                                                         | Expected Result                           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| No `open_clip` import               | `grep -r "open_clip" src/ tests/ scripts/ --include="*.py"`                                                                                                                                                                                                                  | 0 matches                                 |
| No CLIP encoder class               | `grep -r "CLIPEncoder" src/ tests/ scripts/ --include="*.py"`                                                                                                                                                                                                                | 0 matches                                 |
| No text prompt logic                | `grep -r "SEMANTIC_INTENTS\|text_prompt\|encode_text" src/ --include="*.py"`                                                                                                                                                                                                 | 0 matches                                 |
| No cosine similarity head           | `grep -r "cosine_similarity\|logit_scale" src/ --include="*.py"`                                                                                                                                                                                                             | 0 matches                                 |
| MediaPipe keypoints pass through    | `python -c "from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult; import numpy as np; r = HandDetectionResult(bbox=(0,0,10,10), landmarks=np.zeros((21,3)), handedness='Right', confidence=0.9, timestamp=0); print(r.landmarks.shape)"` | `(21, 3)`                                 |
| ViT-B/16 loads cleanly              | `python -c "import timm; m = timm.create_model('vit_base_patch16_224', pretrained=False); print('OK')"`                                                                                                                                                                      | `OK`                                      |
| GCNConv loads cleanly               | `python -c "from torch_geometric.nn import GCNConv; print('OK')"`                                                                                                                                                                                                            | `OK`                                      |
| Package installs without CLIP       | `pip install -e . 2>&1 \| grep -i "clip"`                                                                                                                                                                                                                                    | 0 matches                                 |
| Test suite does not error on import | `python -m pytest tests/ --collect-only 2>&1`                                                                                                                                                                                                                                | No `ImportError` or `ModuleNotFoundError` |
| Evaluation split data intact        | `ls data/hagrid_subset/`                                                                                                                                                                                                                                                     | 8 gesture class directories present       |

---

_This strategy targets surgical, irreversible removal of 52 files and 4 directories. No archive folders are created — deletion is final. The result is a focused, unambiguous codebase where every present file contributes to Architecture A's implementation._
