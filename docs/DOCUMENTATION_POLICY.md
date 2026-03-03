# Documentation Policy

**Version:** 1.0  
**Date:** February 12, 2026  
**Status:** ENFORCED - All documentation must comply

---

## Purpose

This policy establishes strict rules for creating, maintaining, and organizing documentation to prevent sprawl, redundancy, and conflicting information. Adherence is mandatory for all contributors.

---

## Core Principles

### 1. Single Source of Truth

**Rule:** One canonical source per topic. No duplicate explanations.

**Enforcement:**

- Before creating new documentation, search existing docs
- If topic exists, update existing doc or reference it
- Multiple files on same topic = violation

**Canonical Sources:**

| Topic                    | Canonical Source                             |
| ------------------------ | -------------------------------------------- |
| System Architecture      | `docs/architecture/SEMANTIC_SYSTEM_...`      |
| API Contracts            | Code docstrings                              |
| Installation             | `docs/installation/INSTALL.md`               |
| Progress Tracking        | `docs/development/PROJECT_PROGRESS.md`       |
| Implementation History   | `docs/development/work-logs/YYYY-MM-DD_*.md` |
| Operational Workflows    | `docs/operations/` (guides only)             |
| Mathematical Foundations | `docs/architecture/CLIP_MATHEMATICAL_...`    |

---

### 2. No Duplicate Documentation

**Rule:** Never document the same change in multiple files.

**Examples of Violations:**

- ❌ Creating "Interface Drift Report" + "Stabilization Complete Report" describing same fixes
- ❌ Repeating test results in 3 different reports
- ❌ Copy-pasting API examples across multiple docs

**Correct Approach:**

- ✅ Document change once in work-log
- ✅ Reference work-log from other docs if needed
- ✅ Summarize (don't duplicate) in progress tracker

---

### 3. Clear Document Categories

**Rule:** Each document belongs to exactly one category.

**Directory Structure:**

```
docs/
├── architecture/           # Design specifications (timeless)
│   ├── SEMANTIC_SYSTEM_ARCHITECTURE.md
│   └── CLIP_MATHEMATICAL_SPECIFICATION.md
│
├── installation/           # Setup guides
│   └── INSTALL.md
│
├── operations/             # Operational workflows & procedures
│   ├── PACKAGING_REFACTOR_REPORT.md (technical deep-dive)
│   └── PRODUCTION_FINALIZATION_GUIDE.md (workflow guide)
│
├── development/
│   ├── PROJECT_PROGRESS.md              # Milestone tracker (high-level)
│   └── work-logs/                       # Chronological history (detailed)
│       ├── 2026-02-10_clip-integration.md
│       └── 2026-02-12_production-pipeline-stabilization.md
│
└── archive/                # Historical materials (not actively referenced)
    ├── research/
    ├── academic/
    ├── deprecated/
    └── training/
```

---

## Document Type Definitions

### Architecture Documents

**Purpose:** Define system design, components, and principles (timeless)

**Location:** `docs/architecture/`

**Rules:**

- ✅ Describe "what" and "why", not "how to"
- ✅ Focus on design decisions, not implementation details
- ✅ Update when architecture changes (rarely)
- ❌ No implementation history or changelogs
- ❌ No step-by-step instructions

**Examples:**

- `SEMANTIC_SYSTEM_ARCHITECTURE.md` - 5-layer system design
- `CLIP_MATHEMATICAL_SPECIFICATION.md` - Probability constraints

---

### Work-Logs

**Purpose:** Chronological record of implementation changes (detailed history)

**Location:** `docs/development/work-logs/`

**Naming:** `YYYY-MM-DD_brief-description.md`

**Rules:**

- ✅ One log per day/task
- ✅ Document: context, problem, solution, results, lessons learned
- ✅ Append to existing log if same day/task
- ❌ No repeating information from earlier logs
- ❌ No copying entire sections from other documents

**Structure:**

```markdown
# Work Log: [Brief Title]

**Date:** YYYY-MM-DD
**Phase:** [Phase Name]
**Status:** ✅ Complete / ⏳ In Progress

## Context

[Why this work was needed]

## Problem/Objective

[What needed to be solved/implemented]

## Solution

[What was done]

## Results

[Validation, metrics, outcomes]

## Lessons Learned

[What went well, what didn't, prevention strategies]

## References

[Links to code, related logs, documentation]
```

---

### Progress Tracker

**Purpose:** High-level milestone tracking (summary only)

**Location:** `docs/development/PROJECT_PROGRESS.md`

**Rules:**

- ✅ List phases and their status
- ✅ Reference work-logs for details
- ✅ Keep "Recent Milestones" to last 8 entries
- ❌ No detailed technical explanations
- ❌ No full validation results (summarize only)
- ❌ No reproducing work-log content

**Update Frequency:** After each major milestone

---

### Operational Guides

**Purpose:** Workflows, procedures, how-to guides

**Location:** `docs/operations/`

**Rules:**

- ✅ Focus on "how to use" the system
- ✅ Step-by-step instructions
- ✅ Reference code, don't duplicate it
- ❌ No implementation history
- ❌ No validation reports (those go in work-logs)

**Valid Types:**

- Workflow guides (e.g., calibration pipeline)
- Technical deep-dives (e.g., packaging refactor analysis)
- Troubleshooting guides

---

### README Files

**Purpose:** User-facing overview and quick start

**Rules:**

- ✅ Keep concise (< 300 lines ideal)
- ✅ High-level system description
- ✅ Installation (reference INSTALL.md)
- ✅ Core commands
- ❌ No internal audit history
- ❌ No detailed architecture (reference architecture doc)
- ❌ No validation details

---

## Enforcement Rules

### Before Creating New Documentation

**Checklist:**

1. [ ] Search existing docs for topic
2. [ ] Check if work-log entry exists for this date
3. [ ] Verify category (architecture/operations/work-logs)
4. [ ] Confirm no duplicate explanations
5. [ ] If updating existing doc, check git history for conflicts

**If Topic Exists:**

- Option A: Update existing document
- Option B: Reference existing doc with additional context
- Option C: Archive old doc if superseded

**Never:** Create new doc with overlapping content

---

### Updating Existing Documentation

**Rules:**

1. **Update date at top of document**
2. **Add note about what changed** (if significant)
3. **Check all references** to updated content
4. **Update progress tracker** if milestone-related

**Breaking Changes:**

If update invalidates other docs:

1. Update all affected references
2. Archive superseded docs (don't delete)
3. Add note in archived doc pointing to replacement

---

### No Ad-Hoc Reports

**Rule:** Validation results and implementation summaries go in work-logs, not standalone reports.

**Violations (Examples from Feb 12):**

- ❌ INTERFACE_DRIFT_REPORT.md (should be work-log section)
- ❌ PRODUCTION_CLIP_VALIDATION_REPORT.md (should be work-log section)
- ❌ PRODUCTION_STABILIZATION_COMPLETE.md (should be work-log)

**Correct Approach:**

```markdown
# Work Log: Production Pipeline Stabilization

## Phase 1: Root Cause Analysis

[Content from Interface Drift Report]

## Phase 6: CLIP Validation

[Content from CLIP Validation Report]

## Summary

[Content from Stabilization Complete Report]
```

**Exception:** Deep technical analyses with lasting reference value (e.g., PACKAGING_REFACTOR_REPORT.md) may remain in operations/.

---

### Archiving Policy

**When to Archive:**

- Document superseded by newer version
- Research materials no longer actively referenced
- Deprecated approaches (e.g., pre-refactor training guides)
- Conflicting architecture docs (old vs new)

**How to Archive:**

1. Move to appropriate `docs/archive/` subdirectory
2. Add note at top: `**ARCHIVED:** [Reason] | Superseded by: [Link]`
3. Create/update README in archive directory
4. Update references in active docs

**Never Delete:**

- Historical context is valuable
- Git history preserves all content anyway

---

## Specific Rules by Scenario

### Scenario 1: Implementing New Feature

**Documentation:**

1. **During Implementation:** Add entry to work-log (context, approach, challenges)
2. **After Completion:** Update progress tracker (milestone status)
3. **If Needed:** Update operational guide (how to use new feature)

**Do NOT:**

- Create "Feature Implementation Report"
- Create "Feature Validation Report"
- Create "Feature Summary Document"

---

### Scenario 2: Bug Fix

**Documentation:**

1. **Add to work-log:** Problem, root cause, fix, verification
2. **If Major:** Update progress tracker with milestone

**Do NOT:**

- Create "Bug Fix Report"
- Create separate validation document

---

### Scenario 3: Performance Optimization

**Documentation:**

1. **Add to work-log:** Baseline, changes, benchmarks, results
2. **If Significant:** Update architecture doc (if design changed)

**Do NOT:**

- Create "Performance Report"
- Create "Benchmark Results" doc

---

### Scenario 4: Refactoring

**Documentation:**

1. **Major Refactor:** Operations doc with technical analysis (e.g., packaging refactor)
2. **Implementation:** Work-log with changes and validation
3. **Update:** Progress tracker with milestone

**Do NOT:**

- Create multiple reports for same refactor
- Duplicate API changes across multiple docs

---

## Maintenance Schedule

### Weekly Review

**Owner:** Project maintainer

**Tasks:**

1. Check for duplicate documentation
2. Verify work-logs are up to date
3. Archive outdated materials
4. Update progress tracker

---

### Monthly Audit

**Owner:** Project maintainer

**Tasks:**

1. Review documentation structure
2. Consolidate overlapping content
3. Update archive READMEs
4. Verify all references valid

---

## Violation Consequences

**Minor Violations:**

- Reminder to follow policy
- Assist with consolidation

**Major Violations:**

- Documentation PR rejected
- Required: Consolidate before merge

**Repeated Violations:**

- Policy review and training

---

## Examples

### ✅ CORRECT: Work-Log Entry

```markdown
# Work Log: CLIP Integration

**Date:** 2026-02-10

## Context

Phase 1-2 implementation per architecture spec.

## Implementation

- Created CLIPEncoder (512D embeddings)
- Created SemanticIntentExtractor (19 intents)
- Validated latency <100ms

## Results

- 22/22 tests passing
- 75.6ms mean latency

## References

- src/semantic_hgr/representation/clip_encoder.py
- tests/test_clip_encoder.py
```

---

### ❌ INCORRECT: Multiple Reports

```
docs/
├── CLIP_INTEGRATION_REPORT.md          # Violation
├── CLIP_VALIDATION_REPORT.md           # Violation
├── CLIP_IMPLEMENTATION_SUMMARY.md      # Violation
└── CLIP_PERFORMANCE_ANALYSIS.md        # Violation
```

**Problem:** 4 files describing same implementation.

**Fix:** Consolidate into single work-log entry.

---

### ✅ CORRECT: Reference Don't Duplicate

**In Progress Tracker:**

```markdown
## Recent Milestones

- **2026-02-12**: ✅ Production stabilization complete
  ([details](work-logs/2026-02-12_production-pipeline-stabilization.md))
```

**Not:**

```markdown
## Recent Milestones

- **2026-02-12**: ✅ Production stabilization complete
  - Fixed CLIPEncoder API mismatches
  - Refactored 4 scripts
  - Created Makefile
  - [full duplication of work-log content...]
```

---

## Summary

**Golden Rules:**

1. **One topic, one source** - No duplicates
2. **Work-logs for history** - Chronological implementation records
3. **Progress tracker for status** - High-level milestones only
4. **No ad-hoc reports** - Consolidate into work-logs
5. **Archive don't delete** - Preserve history

**Enforcement:**

- Check existing docs before creating new ones
- Reference existing docs instead of duplicating
- Consolidate redundant docs immediately
- Update progress tracker after milestones

**Result:** Clean, navigable, maintainable documentation structure with zero redundancy.

---

**Policy Version:** 1.0  
**Effective Date:** February 12, 2026  
**Review Date:** March 12, 2026
