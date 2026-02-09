# Legacy Documentation Archive

**Archive Date**: February 10, 2026  
**Reason**: Phase 0 cleanup - Repository realignment with semantic architecture

---

## Purpose of This Archive

This directory contains **historical documentation** that presented the project as a gesture classifier rather than a semantic understanding system. These documents are preserved for historical context but are **no longer authoritative**.

---

## What Happened

### Before Phase 0 (Classification-Focused)

The repository contained contradictory documentation:

- Main README described the project as "gesture classification system"
- Documentation mixed classifier terminology with semantic aspirations
- Architecture diagrams showed ViT classifier as primary component
- Implementation focused on supervised classification training

### After Phase 0 (Semantic-Focused)

The repository now aligns with its **true objective**:

- Project is a **semantic gesture understanding system**
- Architecture uses **CLIP vision-language models** for semantic intent extraction
- Implementation path is **research-backed** and feasible
- Zero-shot capability through text descriptions (not retraining classifiers)

### The Correction

**Previous Misconception**:

> "Hand Gesture Recognition System using Vision Transformers for gesture classification"

**Actual Objective**:

> "Semantic Gesture Understanding System that maps gestures to commands through explicit semantic intent representation"

---

## Archived Documents

### classification-approach/

Contains old documentation that framed the project as a classifier:

- **README-old.md**: Original README describing "gesture classification system"
- **START_HERE.md**: Quick start guide focused on ViT classifier training
- **DEVELOPMENT_GUIDE.md**: Development workflow mixing classification and semantic terminology

**Why Archived**: These documents presented **gesture classification** as the end goal, when it should only be a baseline for comparison. The semantic architecture supersedes this approach.

### Root-Level Meta-Documentation

- CLEANUP_SUMMARY.md
- TIMESTAMP_UPDATE_SUMMARY.md
- IMPORT_ISSUE_FIX.md
- LOG_ROTATION_STRATEGY.md
- MIGRATION_CHECKLIST.md
- NAMING_CONVENTION.md
- PACKAGE_MANAGEMENT.md
- QUICK_REFERENCE.md

**Why Archived**: Temporary meta-documentation from iterative development. Not relevant to the semantic architecture. Cluttered root directory without providing authoritative guidance.

---

## Authoritative Documentation

**DO NOT USE ARCHIVED DOCUMENTS FOR IMPLEMENTATION**.

The single source of truth is:

📄 **[/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)**

This document defines:

- Project objective (semantic understanding)
- Five-layer architecture
- Research foundation (CLIP, SentenceTransformers)
- Technology selection
- Phased implementation plan (Phase 1-4)
- Repository structure
- Success metrics

---

## Historical Value

These documents are preserved because they show:

1. **Evolution**: How the project shifted from classification to semantic understanding
2. **Baseline**: ViT classifier training (85-90% accuracy on HaGRID) serves as comparison baseline
3. **Lessons**: Why classifier-only approaches are insufficient for semantic gesture control

---

## If You're New to This Project

**DO THIS**:

1. Read [/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md](/docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
2. Understand five-layer semantic architecture
3. Follow Phase 1 implementation plan

**DO NOT**:

- Use archived documentation as implementation guide
- Build gesture classifiers without semantic layer
- Hard-code gesture-to-command mappings

---

## Questions About This Archive?

See `work-logs/2026-02-10_phase0-cleanup.md` for complete rationale behind Phase 0 cleanup decisions.

**Summary**: These documents contradicted the semantic architecture and have been superseded by authoritative specification.
