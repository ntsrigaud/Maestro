# Archive: Pre-Fix CLIP Validation Results

**Archive Date:** February 11, 2026  
**Reason:** Mathematical Error - Missing Softmax Transformation  
**Status:** ❌ SUPERSEDED - DO NOT USE FOR CURRENT SYSTEM

---

## Why These Results Were Archived

This directory contains validation results and analysis documents created before the critical softmax correction on February 11, 2026. **These results are mathematically incorrect** and should not be used for evaluating the current system.

### Mathematical Error

**Problem:** The code returned raw **logits** as "confidence" scores without softmax normalization.

**Impact:** "Confidence" values exceeded 1.0 (e.g., 509.02), violating probability axioms:

- ❌ Values > 1.0 (not valid probabilities)
- ❌ Do not sum to 1.0 across intents
- ❌ Cannot be interpreted as probabilities

**Example (INCORRECT):**

```
Top intent: scroll_up, confidence: 509.02
Mean confidence: 402.15
```

These are **logits**, not probabilities!

### Correction Applied

**Date:** February 11, 2026  
**Fix:** Added softmax transformation in `clip_intent.py` (lines 410-412, 525-527)

**Formula:**

```python
# Before (WRONG):
logits = cosine_sim * logit_scale / temperature
return logits  # ❌ Can be > 1.0

# After (CORRECT):
logits = cosine_sim * logit_scale / temperature
exp_logits = np.exp(logits - logits.max())
probabilities = exp_logits / exp_logits.sum()  # ✅ Valid [0,1], sum=1.0
return probabilities
```

### Corrected Results

**Current System (February 11, 2026):**

- Top-1 Probability: 0.960 (96.0%) ✅
- Success Rate: 98/100 (≥0.80 threshold) ✅
- All probabilities ∈ [0, 1] ✅
- Probabilities sum to 1.0 ✅

**See:** [Mathematical Validation Report](../work-logs/2026-02-11_mathematical-validation-report.md)

---

## Archived Documents

### 1. Phase 1-2 Confidence Fix (Superseded)

**File:** [2026-02-11_phase1-2-confidence-fix.md](2026-02-11_phase1-2-confidence-fix.md)  
**Original Location:** `docs/work-logs/2026-02-11_phase1-2-confidence-fix.md`  
**Archive Reason:** Contains incorrect metrics (confidence values >1.0)

**Contents:**

- Root cause analysis (correct identification of missing softmax)
- Incorrect validation results (logits reported as confidence)
- Implementation notes (partially correct, but metrics invalid)

**Superseded By:**

- [Mathematical Validation Report](../work-logs/2026-02-11_mathematical-validation-report.md) - Correct metrics and validation
- [CLIP Mathematical Specification](../CLIP_MATHEMATICAL_SPECIFICATION.md) - Authoritative formula

**What's Still Valid:**

- ✅ Root cause analysis (identified missing softmax)
- ✅ Code locations (lines 398-420, 515-531)
- ✅ Parameter values (logit_scale≈100, temperature=0.07)

**What's Invalid:**

- ❌ All "confidence" metrics (509.02, 402.15, etc.)
- ❌ Interpretation as probabilities
- ❌ Threshold analysis based on logits

**Warning Notice:** Document contains "⚠️ SUPERSEDED" notice at top directing readers to corrected reports.

---

## When to Use These Documents

### ❌ DO NOT USE FOR:

- Current system evaluation
- Production deployment decisions
- Probability calibration analysis
- Threshold tuning
- Performance reporting

### ✅ CAN USE FOR:

- Understanding historical bug
- Learning what NOT to do (educational)
- Tracing system evolution
- Debugging similar issues in other projects

---

## Corrected Documentation

Use these documents instead:

1. **Current Validation Results:**
   - [Mathematical Validation Report](../work-logs/2026-02-11_mathematical-validation-report.md)
   - Contains corrected metrics with valid probabilities

2. **Mathematical Specification:**
   - [CLIP Mathematical Specification](../CLIP_MATHEMATICAL_SPECIFICATION.md)
   - Authoritative formula derivation and implementation

3. **Unit Test Validation:**
   - [Phase 1: Unit Test Hardening](../work-logs/2026-02-11_phase1-unit-test-hardening.md)
   - 10/10 tests passing, probability constraints validated

4. **Documentation Update:**
   - [Phase 2: Documentation Update](../work-logs/2026-02-11_phase2-documentation-update.md)
   - READMEs updated with corrected mathematical pipeline

---

## Historical Context

### Timeline

**2026-02-10:** Initial CLIP integration completed (Phase 1-2)  
**2026-02-11 (Early):** User identified confidence values > 1.0 as mathematically invalid  
**2026-02-11 (Mid):** Root cause analysis → Missing softmax transformation  
**2026-02-11 (Mid):** Softmax correction applied to `clip_intent.py`  
**2026-02-11 (Late):** Validation confirmed: all probabilities now valid  
**2026-02-11 (Late):** Old documents archived, new validation report created

### Lesson Learned

**Principle:** "Bigger numbers are not better numbers. Correct math is better math."

**Takeaway:** Always validate that probability scores:

1. Are within [0, 1]
2. Sum to 1.0 across all classes
3. Come from proper softmax normalization

**Prevention:** Unit tests added (`test_probability_validation.py`) to catch this type of error automatically.

---

## Archive Maintenance

**Review Schedule:** Annual (February each year)  
**Deletion Policy:** Retain indefinitely for historical reference  
**Access:** Read-only, no modifications  
**Warnings:** All documents have "SUPERSEDED" notices at top

---

**Archive Created By:** Neil Taison Rigaud  
**Archive Date:** February 11, 2026  
**Last Updated:** February 11, 2026
