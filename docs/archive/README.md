# Archive Overview

**Last Updated:** February 12, 2026

This directory contains historical, deprecated, and archived documentation from the Semantic HGR System project.

---

## Archive Structure

```
archive/
├── deprecated/          # Superseded architectures and components
├── training/            # Training materials (not used in current system)
├── research/            # Research assessments and dataset documentation
├── academic/            # Academic materials and course content
├── legacy/              # Legacy implementation documentation
├── clip_pre_fix_validation/  # Historical pre-softmax fix documents
└── meeting-notes/       # Project meeting notes (2025-09 to 2026-01)
```

---

## Subdirectory Descriptions

### `deprecated/`

**Purpose:** Superseded architectural designs and components

**Contents:**

- `Project-Architecture/` - Component-based architecture (replaced by semantic architecture)
- `Recognition-Engine/` - LSTM/ViT workflow documentation (replaced by CLIP zero-shot)
- `setup.py.deprecated` - Old packaging configuration

**Archived:** February 12, 2026  
**Reason:** System redesigned with semantic architecture (see [SEMANTIC_SYSTEM_ARCHITECTURE.md](../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md))

---

### `training/`

**Purpose:** Model training and fine-tuning documentation

**Contents:**

- `MODEL_TRAINING_GUIDE.md` - ViT classifier training guide
- `QUICK_REFERENCE_TRAINING.md` - Training command shortcuts
- `TRAINING_ENHANCEMENT_SUMMARY.md` - Training optimization strategies
- `TRAINING_SETUP_SUMMARY.md` - Training environment setup

**Archived:** February 12, 2026  
**Reason:** Current system uses **pretrained CLIP** with zero-shot semantic understanding. No training or fine-tuning is performed.

**Note:** These materials may be useful if future work involves fine-tuning CLIP or training custom models.

---

### `research/`

**Purpose:** Research materials, dataset assessments, and exploratory documentation

**Contents:**

- `HaGRID_DATASET_ASSESSMENT.md` - HaGRID dataset analysis
- `HAGRID_DOWNLOAD_GUIDE.md` - Dataset download instructions
- `HAGRID_DOWNLOAD_SOLUTIONS.md` - Dataset download troubleshooting
- `TECHNICAL_RESEARCH_AUDIT.md` - Technical research survey
- `core-action-set.md` - Action vocabulary research
- `graduation_project_index.md` - Academic project index

**Archived:** February 12, 2026  
**Reason:** Research materials from project planning phase. HaGRID dataset was explored for classification approach (no longer used).

**Note:** Useful for understanding project research history and considered approaches.

---

### `academic/`

**Purpose:** Academic course materials, theoretical explanations, and proposals

**Contents:**

- `Imaging/` - Computer vision theory (imaging formation, lenses, pinhole projection)
- `Research-Notes/` - Historical research notes
- `Topics/` - Project proposals and contribution explanations

**Archived:** February 12, 2026  
**Reason:** Educational and theoretical materials not directly related to current production system.

**Note:** Preserved for educational reference and project context.

---

### `legacy/`

**Purpose:** Legacy implementation documentation and migration history

**Status:** Pre-existing archive (already organized)

**Contents:**

- Classification approach documentation
- Import issue fixes
- Naming conventions
- Package management strategies
- Migration checklists

**Note:** Documents transitions from earlier system iterations.

---

### `clip_pre_fix_validation/`

**Purpose:** Historical validation documents from pre-softmax confidence fix

**Status:** Pre-existing archive

**Archived:** February 11, 2026  
**Reason:** Historical artifact documenting the confidence calculation fix (pre-softmax migration).

---

### `meeting-notes/`

**Purpose:** Project meeting notes and milestone discussions

**Status:** Pre-existing archive

**Contents:** 18 meeting notes (2025-09-23 to 2026-01-23)

**Note:** Valuable historical context for project decisions and evolution.

---

## Archival Policy

**See:** [ARCHIVE_POLICY.md](../../ARCHIVE_POLICY.md) at repository root for complete archival guidelines.

**Key Principles:**

1. **Never delete** - Archive, don't destroy
2. **Clear disclaimers** - Mark archived documents with status headers
3. **Preserve context** - Explain why material was archived
4. **Link alternatives** - Point to current documentation where applicable

---

## Current Documentation

For active, production documentation, see:

- **Architecture:** [docs/architecture/](../architecture/)
- **Installation:** [docs/installation/](../installation/)
- **Operations:** [docs/operations/](../operations/)
- **Development:** [docs/development/](../development/)
- **Research:** [docs/research/](../research/) (active research references)

**Documentation Index:** [docs/README.md](../README.md)

---

**This archive preserves complete project history while maintaining a clean production documentation structure.**
