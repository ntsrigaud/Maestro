# Deprecated Documentation

**Archived:** February 12, 2026  
**Status:** SUPERSEDED

---

## Contents

### `Project-Architecture/`

Component-based system architecture that predated the semantic reboot.

**Superseded by:** [SEMANTIC_SYSTEM_ARCHITECTURE.md](../../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md)

**Conflict:** Described different 5-layer system with:

- Input Manager
- Recognition Engine (LSTM/ViT-based)
- Action Mapper
- OS Controller

Current system uses semantic architecture with CLIP-based zero-shot understanding.

---

### `Recognition-Engine/`

Documentation for LSTM/ViT classification workflow.

**Superseded by:** CLIP semantic understanding (no classification model)

**Key Document:** `LSTM_VIT_WORKFLOW.md` - Described training and inference pipeline for gesture classification

---

### `setup.py.deprecated`

Old Python packaging configuration file.

**Superseded by:** `pyproject.toml` with unified `semantic_hgr` namespace

**See:** [PACKAGING_REFACTOR_REPORT.md](../../operations/PACKAGING_REFACTOR_REPORT.md) for details

---

## Why Archived

These documents described a **component-based classification system** that was redesigned as a **semantic understanding system**:

| Old Approach        | Current Approach        |
| ------------------- | ----------------------- |
| ViT classification  | CLIP zero-shot semantic |
| 18 gesture classes  | Open vocabulary intents |
| Trained model       | Pretrained CLIP         |
| Hard-coded mappings | Semantic similarity     |

---

## Historical Value

Preserved for:

- Understanding project evolution
- Design decision rationale
- Potential future reference

**Do not use for current system implementation.**
