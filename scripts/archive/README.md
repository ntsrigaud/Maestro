# Scripts Archive

**Archive Date**: 2026-02-10  
**Reason**: Legacy script purge during Phase 1 corrective enforcement

---

## Context

These scripts were archived because they relate to the **classification approach** (ViT-B/16 training, dataset utilities, legacy demos), which contradicts the semantic gesture understanding architecture.

---

## Archive Structure

### classification/

Scripts for ViT classifier training and inference (archived approach):

- `demo_gesture_recognition.py` - ViT classifier real-time demo
- `real_time_inference.py` - ViT + LSTM temporal classification
- `verify_clean_baseline.py` - ViT baseline verification
- `test_reproducibility.py` - Reproducibility tests for classifier training
- `upload_model_to_huggingface.py` - HuggingFace model upload for classifiers
- `build_no_gesture2_class.py` - No-gesture class builder
- `build_no_gesture2_simple.py` - Simplified no-gesture builder

### dataset-utils/

Dataset download and organization scripts (not needed for Phase 1):

- `download_*.py` - Various dataset downloaders (HaGRID, COCO, CelebA)
- `organize_hagrid_dataset.py` - Dataset organization
- `create_train_val_test_splits.py` - Train/val/test splitting
- `audit_dataset_balance.py` - Class balance auditing
- `capture_hand_samples.py` - Hand sample capture utility

### legacy-demos/

Old test scripts and demos (superseded):

- `test_*.py` - Various unit tests for old architecture
- `demo_timestamp_usage.py` - Timestamp usage examples
- `setup_*.sh` - Setup scripts for legacy experiments
- `verify_*.py` - Old verification scripts
- `run_gesture_demo.sh` - Shell script for old demos

### Miscellaneous

- `GESTURE_RECOGNITION_DEMO.md` - Demo documentation for classifier
- `ORGANIZE_DATASET_README.md` - Dataset organization guide
- `environment.yml` - Misplaced environment file (duplicate)

---

## Active Scripts (Retained)

Only 6 scripts remain in `/scripts/`:

1. `validate_input_layer.py` - MediaPipe validation (Phase 1 requirement)
2. `validate_hand_detection.py` - Hand detection validation
3. `demo_hand_detection.py` - Hand detection demo
4. `verify_install.py` - Installation verification
5. `README.md` - Scripts documentation
6. `archive/` - This archive directory

---

## Justification

**Why Archived**:

- Classification scripts contradict semantic architecture
- Dataset utilities not needed for Phase 1 (input validation only)
- Legacy demos superseded by new validation scripts
- Duplicated environment files create confusion

**Why Not Deleted**:

- Historical reference for baseline comparison
- Potential future use for performance benchmarking
- Archive preserves project evolution

---

## Future Use

These scripts may be useful for:

- Comparing semantic system performance to classification baseline
- Understanding project evolution
- Extracting insights from previous training experiments

**Not for**: Current development or Phase 1 work
