# Notebooks Archive

**Archive Date**: February 10, 2026  
**Reason**: Phase 0 cleanup - Archived classification-focused and aspirational notebooks

---

## Purpose of This Archive

This directory contains **historical notebooks** that either:

1. Implemented gesture **classification** (not semantic understanding)
2. Described **aspirational dual-encoder systems** that were never implemented
3. Contained **experimental code** not aligned with the semantic architecture

---

## Archived Notebooks

### finetune_vit_hagrid.ipynb

**Original Purpose**: Train ViT-B/16 classifier on HaGRID dataset (18 gesture classes)

**Results**:

- 85-90% classification accuracy
- Model published: `neilrigaud/hagrid-vit-gesture`

**Why Archived**:

- This is a **baseline classifier**, not the semantic system
- Useful for **comparison**, but not the target architecture
- New architecture uses **CLIP** (vision-language), not custom ViT classifiers

**Historical Value**:

- Shows classification baseline performance
- Demonstrates ViT training pipeline
- Comparison point for semantic approach

---

### lstm_train_vit_features.ipynb

**Original Purpose**: Experimental temporal modeling with LSTM

**Status**: Incomplete, untested

**Why Archived**:

- Expects different dataset structure (ViT features, not raw video)
- Not validated or tested
- New architecture uses **VideoCLIP/X-CLIP** for temporal understanding (Phase 3)

**Historical Value**:

- Shows early temporal modeling exploration
- Demonstrates why temporal requires proper architecture (not bolt-on LSTM)

---

### SEMANTIC_GESTURE_RESEARCH.md

**Original Purpose**: Research document describing dual-encoder contrastive learning approach

**Content**:

- ALBEF-inspired architecture
- Contrastive loss (InfoNCE)
- Custom projection layers
- Custom training loop

**Why Archived**:

- Describes system that was **never implemented**
- Proposed **custom architecture** when production-ready models (CLIP) exist
- Overly complex for Phase 1 goals

**Historical Value**:

- Shows research direction that led to simpler CLIP approach
- Demonstrates why custom dual-encoders aren't necessary (CLIP already trained)

---

### GESTURE_SEMANTICS_ENHANCEMENT.md

**Original Purpose**: Document gesture-text pair expansion efforts

**Why Archived**:

- Describes data preparation for **unimplemented** dual-encoder system
- Not relevant to CLIP approach (CLIP uses pre-trained embeddings)

---

### QUICK_START_SEMANTIC.md

**Original Purpose**: Quick start guide for dual-encoder training

**Why Archived**:

- Describes **unimplemented** training pipeline
- Not accurate guide for current architecture

---

### SEMANTIC_IMPROVEMENT_GUIDE.md

**Original Purpose**: Improvement suggestions for dual-encoder system

**Why Archived**:

- System was never implemented, so improvements are moot

---

### NO_GESTURE2_INTEGRATION.md

**Original Purpose**: Document no_gesture2 class experiments

**Why Archived**:

- Experimental 19th class not in final 18-class model
- Published model (`neilrigaud/hagrid-vit-gesture`) has 18 classes, not 19
- Experimental work that didn't make it to production

---

### TRAINING_COMPARISON.md

**Original Purpose**: Compare simple vs improved classification heads

**Why Archived**:

- Pure classification comparison
- Not relevant to semantic architecture

---

### gesture_semantics.py

**Original Purpose**: Semantic gesture utilities (unclear functionality)

**Why Archived**:

- Not integrated into any pipeline
- Misplaced production code in notebooks directory

---

### inference_gesture.py

**Original Purpose**: GestureClassifier implementation for ViT model inference

**Status**: Complete, working (198 lines)

**Why Archived**:

- **Classification-only** - no semantic understanding
- Should be in `src/`, not `notebooks/`
- New architecture replaces this with **CLIP-based intent extraction**

**Historical Value**:

- Shows working inference code structure
- Demonstrates preprocessing pipeline
- Reference for implementation patterns

---

## What Replaces These?

### Phase 1 Implementation (New)

New notebooks in `notebooks/phase1_semantic_foundation/`:

1. **01_clip_exploration.ipynb** - CLIP zero-shot gesture understanding
2. **02_intent_vocabulary_design.ipynb** - Designing semantic intent vocabulary
3. **03_intent_matching.ipynb** - Cosine similarity intent matching
4. **04_command_resolution.ipynb** - Semantic → command mapping
5. **05_end_to_end_pipeline.ipynb** - Full pipeline integration
6. **06_evaluation.ipynb** - Zero-shot accuracy, latency, safety validation

### Key Differences

| Archived Approach                 | New Approach (Phase 1)           |
| --------------------------------- | -------------------------------- |
| Train custom dual-encoder         | Use pre-trained CLIP             |
| Collect gesture-text pairs        | Use text descriptions only       |
| Contrastive learning from scratch | Zero-shot via CLIP embeddings    |
| Custom projection layers          | Direct CLIP text/vision encoders |
| Supervised classification         | Semantic similarity matching     |

---

## Authoritative Documentation

**DO NOT USE ARCHIVED NOTEBOOKS FOR IMPLEMENTATION**.

Follow:

- 📄 [/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md) - Architecture specification
- 📓 `notebooks/phase1_semantic_foundation/` - New implementation notebooks (when created)

---

## If You're Implementing Phase 1

**DO THIS**:

1. Read [/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
2. Use CLIP (OpenCLIP) for vision-language embeddings
3. Follow Phase 1 week-by-week plan

**DO NOT**:

- Train custom dual-encoders (CLIP already trained)
- Use classification notebooks as implementation guide
- Implement LSTM temporal modeling (Phase 3, not Phase 1)

---

## Questions About This Archive?

See `work-logs/2026-02-10_phase0-cleanup.md` for complete rationale.

**Summary**: These notebooks describe unimplemented aspirational systems or classification-only approaches that contradict the semantic architecture.
