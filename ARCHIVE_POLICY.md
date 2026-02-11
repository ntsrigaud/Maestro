# Archive Policy

**Last Updated:** February 12, 2026  
**Status:** Active

---

## Purpose

This document defines the archival policy for the Semantic HGR System documentation. It ensures historical materials are preserved while maintaining a clean, production-ready documentation structure.

---

## Canonical Documentation

The following documents represent the **authoritative technical specification** of the system:

### Primary Sources

1. **`docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md`**  
   Canonical system architecture specification. All architectural decisions defer to this document.

2. **`docs/architecture/CLIP_MATHEMATICAL_SPECIFICATION.md`**  
   Mathematical foundations of CLIP-based semantic understanding.

3. **`docs/installation/INSTALL.md`**  
   Official installation guide reflecting current packaging structure.

4. **`docs/operations/PRODUCTION_FINALIZATION_GUIDE.md`**  
   Production deployment guide (calibration, testing, validation).

5. **`docs/operations/PACKAGING_REFACTOR_REPORT.md`**  
   Technical documentation of Python packaging refactor.

6. **`docs/operations/PRODUCTION_READINESS_ASSESSMENT.md`**  
   System readiness evaluation and production checklist.

7. **`docs/development/PROJECT_PROGRESS.md`**  
   Implementation progress tracker (updated continuously).

### Supporting Documentation

- **`docs/development/work-logs/`** - Implementation work logs (chronological)
- **`docs/research/`** - Research references and literature surveys
- **Root `README.md`** - Project overview and quick-start guide

---

## Archival Categories

### Category 1: Deprecated Architecture (`docs/archive/deprecated/`)

**Criteria:**

- Documents describing system architecture that has been superseded
- Component designs that conflict with canonical architecture
- Implementation guides for approaches no longer used

**Examples:**

- `Project-Architecture/` - Component-based architecture (superseded by semantic architecture)
- `Recognition-Engine/` - LSTM/ViT workflow (superseded by CLIP zero-shot)
- `setup.py.deprecated` - Old packaging approach

**Retention:** Permanent (historical reference)

---

### Category 2: Training Materials (`docs/archive/training/`)

**Criteria:**

- Model training guides
- Fine-tuning documentation
- Dataset preparation instructions for training

**Rationale:**
The current system uses **pretrained CLIP** with zero-shot semantic understanding. No training or fine-tuning is performed. Training materials describe a previous classification-based approach that is no longer part of the system.

**Examples:**

- `MODEL_TRAINING_GUIDE.md` - ViT classifier training
- `TRAINING_ENHANCEMENT_SUMMARY.md` - Training improvements
- `TRAINING_SETUP_SUMMARY.md` - Training environment setup
- `QUICK_REFERENCE_TRAINING.md` - Training shortcuts

**Retention:** Permanent (potential future fine-tuning reference)

---

### Category 3: Research Materials (`docs/archive/research/`)

**Criteria:**

- Dataset assessments
- Technical research audits
- Exploratory analysis documents
- Project planning materials

**Examples:**

- `HaGRID_DATASET_ASSESSMENT.md` - Dataset analysis
- `HAGRID_DOWNLOAD_GUIDE.md` - Dataset download instructions
- `TECHNICAL_RESEARCH_AUDIT.md` - Research survey
- `core-action-set.md` - Action vocabulary research
- `graduation_project_index.md` - Academic project index

**Retention:** Permanent (research reference)

---

### Category 4: Academic Materials (`docs/archive/academic/`)

**Criteria:**

- Course materials
- Theoretical explanations
- Academic proposals
- Educational content

**Examples:**

- `Imaging/` - Computer vision theory (imaging, lenses, projection)
- `Topics/proposal.md` - Project proposal
- `Topics/contribution_explanation.md` - Academic contribution description

**Retention:** Permanent (educational reference)

---

### Category 5: Legacy Implementation (`docs/archive/legacy/`)

**Criteria:**

- Old implementation approaches
- Deprecated code patterns
- Migration checklists from previous system states

**Status:** Already archived (existing directory)

**Examples:**

- Classification-based approach documentation
- Import issue fixes
- Naming conventions from previous iterations
- Package management strategies

**Retention:** Permanent (migration history)

---

### Category 6: Historical Artifacts (`docs/archive/`)

**Criteria:**

- Meeting notes
- Pre-fix validation documents
- Project milestone artifacts

**Status:** Already archived

**Examples:**

- `meeting-notes/` - 18 meeting notes (2025-09 to 2026-01)
- `clip_pre_fix_validation/` - Pre-softmax fix documents

**Retention:** Permanent (project history)

---

## Archival Process

### Step 1: Classification

Determine appropriate category using criteria above.

### Step 2: Disclaimer Addition

Add disclaimer header to archived document:

```markdown
---
**STATUS:** ARCHIVED  
**Date Archived:** [Date]  
**Reason:** [Superseded by X / Obsolete approach / Historical reference]  
**Current Alternative:** [Link to current document if applicable]
---

[Original content below]
```

### Step 3: Move to Archive

Move file/directory to appropriate `docs/archive/` subdirectory.

### Step 4: Update Links

- Update any links in active documentation to point to archived location
- Add context explaining why archived material is referenced
- Consider removing links if no longer relevant

### Step 5: Archive README

Ensure each archive subdirectory has a `README.md` explaining:

- What is archived in this directory
- Why it was archived
- When it was archived
- Links to current alternatives

---

## Anti-Patterns (Do Not Archive)

**❌ Do NOT archive:**

- Current production documentation (see Canonical Documentation)
- Recent work logs (`docs/development/work-logs/2026-*`)
- Active research references (`docs/research/`)
- Test files or validation scripts

**❌ Do NOT delete:**

- Any documentation (preserve history)
- Test results
- Validation outputs

---

## Conflict Resolution

**If two documents describe the same system aspect:**

1. **Identify canonical source** (see Canonical Documentation above)
2. **Archive conflicting document** (see Archival Categories)
3. **Add disclaimer** explaining supersession
4. **Update links** to point to canonical source

**Example:**

- **Conflict:** `Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md` vs `SEMANTIC_SYSTEM_ARCHITECTURE.md`
- **Resolution:** Archive `Project-Architecture/` → `docs/archive/deprecated/`
- **Canonical:** `docs/architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md`

---

## Review Schedule

**Quarterly Review:**

- Identify new documents that should be archived
- Verify archived documents remain accessible
- Update canonical documentation list if system evolution requires

**Trigger Events:**

- Major architectural changes
- Packaging/dependency refactors
- System redesigns
- Model or approach changes

---

## Rationale

**Why archive instead of delete?**

1. **Historical context** - Future developers understand system evolution
2. **Research reference** - Archived approaches may inform future work
3. **Audit trail** - Decision history preserved for retrospective analysis
4. **Academic value** - Project serves as educational case study

**Why strict canonical designation?**

1. **No confusion** - Single source of truth for each system aspect
2. **Confidence** - Developers trust documentation accuracy
3. **Professionalism** - Production systems have clear, authoritative documentation
4. **Maintainability** - Updates go to one place, not scattered duplicates

---

## Archive Structure

```
docs/archive/
├── README.md                           # Archive overview
├── deprecated/                         # Superseded architecture/components
│   ├── README.md
│   ├── Project-Architecture/
│   ├── Recognition-Engine/
│   └── setup.py.deprecated
├── training/                           # Training materials (not used)
│   ├── README.md
│   ├── MODEL_TRAINING_GUIDE.md
│   ├── QUICK_REFERENCE_TRAINING.md
│   ├── TRAINING_ENHANCEMENT_SUMMARY.md
│   └── TRAINING_SETUP_SUMMARY.md
├── research/                           # Research materials
│   ├── README.md
│   ├── HaGRID_DATASET_ASSESSMENT.md
│   ├── HAGRID_DOWNLOAD_GUIDE.md
│   ├── HAGRID_DOWNLOAD_SOLUTIONS.md
│   ├── TECHNICAL_RESEARCH_AUDIT.md
│   ├── core-action-set.md
│   └── graduation_project_index.md
├── academic/                           # Academic materials
│   ├── README.md
│   ├── Imaging/
│   └── Topics/
├── legacy/                             # Legacy implementation (existing)
│   └── [existing files]
├── clip_pre_fix_validation/            # Historical artifacts (existing)
└── meeting-notes/                      # Historical artifacts (existing)
```

---

**This policy ensures the repository maintains production-level documentation standards while preserving complete project history.**
