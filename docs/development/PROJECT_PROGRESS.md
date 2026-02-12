# Progress Tracker

**Project**: Semantic Gesture Understanding System  
**Started**: February 10, 2026  
**Current Phase**: Phase 1 - Input Layer Validation & CLIP Foundation  
**Overall Status**: ✅ Phase 1-1 Complete | ✅ Phase 1-2 Complete | 🎯 Phase 1-3 Ready

---

## Executive Summary

**Architecture**: [../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md](../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md)  
**Latest Milestone**: Research Correction Complete - Static Gesture Track Defined (February 12, 2026)  
**Current Status**: 🔴 **SCOPE REALIGNMENT IN PROGRESS**  
**Next Phase**: Immediate Realignment Action Plan (Static Gesture Track)

**CRITICAL RESEARCH CORRECTION (Feb 12, 2026)**:  
Root cause analysis identified fundamental experimental design flaw: attempted to infer motion-dependent intents (scroll, swipe, zoom) from single RGB frames using vanilla CLIP (no temporal modeling). **Previous calibration study (ECE=0.857) INVALIDATED** due to undefined ground truth for 68% of vocabulary. System scope realigned to static hand pose recognition only.

---

## Phase Overview

| Phase     | Status                  | Progress | Target    |
| --------- | ----------------------- | -------- | --------- |
| Phase 0   | ✅ Complete             | 100%     | Feb 10    |
| Phase 1-1 | ✅ Complete             | 100%     | Week 1    |
| Phase 1-2 | ⚠️ PARTIAL (see below)  | 75%      | Weeks 1-2 |
| Phase 1-3 | 🔴 REALIGNMENT REQUIRED | 0%       | Weeks 2-3 |
| Phase 2   | ⏸️ DEFERRED             | 0%       | Future    |
| Phase 3   | ⏸️ DEFERRED             | 0%       | Future    |

---

## Phase 0: Repository Cleanup (Complete)

**Duration**: 1 day  
**Completed**: February 10, 2026  
**Status**: ✅ Complete

### Deliverables

- ✅ Archived contradictory ViT classifier documentation
- ✅ Established authoritative SEMANTIC_SYSTEM_ARCHITECTURE.md
- ✅ Standardized work-log conventions (docs/work-logs/)
- ✅ Created README with clear project direction
- ✅ Dependency audit (environment.yml, pyproject.toml, setup.py)
- ✅ Legacy script purge (40+ → 6 scripts)

---

## 🔴 CRITICAL RESEARCH CORRECTION (February 12, 2026)

**Status**: ❌ **PREVIOUS CALIBRATION STUDY INVALIDATED**  
**Root Cause Analysis**: [CALIBRATION_ROOT_CAUSE_ANALYSIS.md](../analysis/CALIBRATION_ROOT_CAUSE_ANALYSIS.md)  
**Impact**: High - Requires immediate scope realignment

### Identified Fundamental Flaw

**Problem**: System attempted to evaluate motion-dependent intents (scroll, swipe, zoom) using single-frame RGB inference with vanilla CLIP (ViT-B/32).

**Technical Analysis**:

- **CLIP Architecture**: No temporal modeling (trained on 400M static image-text pairs)
- **Intent Vocabulary**: 13/19 intents (68%) require multi-frame motion information
- **Ground Truth Status**: UNDEFINED for dynamic gestures in single frames
- **Calibration Data**: Synthetic random embeddings with no semantic structure

**Why Previous Results Are Invalid**:

1. **ECE=0.857, Accuracy=4%**: NOT model failure, but task ambiguity
2. **High Confidence (98%)**: Reflects text embedding structure, not visual understanding
3. **Calibration Prerequisites Violated**: No valid ground truth for dynamic intents
4. **Evaluation Impossibility**: Cannot compute accuracy when labels are unknowable

**Literature Evidence**:

- **ActionCLIP** (Wang et al., 2021): Vanilla CLIP achieves 38% accuracy on action recognition without temporal shift modules
- **HaGRID Dataset** (Kapitanov et al., 2024): Contains ONLY static poses (18 classes + background)
- **VideoCLIP** (Xu et al., 2021): Requires 3D convolutions + temporal attention for dynamic gestures

**Mathematical Proof**: For gestures G₁="swipe_left", G₂="swipe_right", ∃ frame t where visual features F(G₁,t) = F(G₂,t), but motion vectors M(G₁) ≠ M(G₂). Single-frame inference cannot distinguish them.

### Invalidated Claims

❌ **RETRACTED**:

- "Calibration study demonstrates model reliability" → FALSE (task was undefined)
- "System understands dynamic gesture intent" → FALSE (requires temporal modeling)
- "Confidence scores are meaningful" → FALSE (computed on synthetic data)
- "96% top-1 probability indicates semantic understanding" → FALSE (only text similarity, not visual grounding)

### Corrected Scope

**ACTIVE SCOPE (Immediate)**:

- ✅ Static hand pose recognition (single-frame inference)
- ✅ CLIP-based vision-language embedding
- ✅ Pose-to-intent association for static gestures
- ✅ Top-K retrieval accuracy on HaGRID-aligned vocabulary

**OUT OF SCOPE (Current System)**:

- ❌ Dynamic gestures (scroll, swipe, zoom, rotate, drag)
- ❌ Temporal intent inference from motion trajectories
- ❌ Probability calibration for motion-dependent actions
- ❌ Multi-frame sequence modeling

**FUTURE WORK (Separate Research Track)**:

- ⏸️ Temporal modeling with ActionCLIP/VideoCLIP (requires 8-16 frame windows)
- ⏸️ Optical flow heuristics for lightweight motion cues
- ⏸️ Real gesture dataset collection for valid calibration

### Postmortem Summary

**What Went Wrong**:  
Architectural contradiction between system capabilities (single-frame) and evaluation vocabulary (motion-dependent). No explicit validation that intent labels were answerable from available information.

**What Was Learned**:  
Zero-shot capability ≠ semantic understanding. CLIP can match text to images, but cannot infer temporal dynamics. Calibration requires task validity first, model training second.

**Path Forward**:  
Realign to static-only vocabulary (9 HaGRID-aligned intents), replace calibration with retrieval metrics, defer dynamic gestures to temporal modeling research.

---

## Phase 1-1: Input Layer Validation (Complete)

**Duration**: Week 1  
**Started**: February 10, 2026  
**Completed**: February 10, 2026  
**Dependencies**: Phase 0 complete ✅  
**Status**: ✅ Complete

### Tasks (9/9 Complete)

- [x] Create MediaPipe validation script
- [x] Execute failure mode tests (empty frames, None input, grayscale)
- [x] Code review: camera capture, hand detection, cropping
- [x] Dependency reconciliation across all config files
- [x] Documentation consolidation
- [x] Work-log directory enforcement
- [x] Automated webcam testing (detection metrics)
- [x] **Visual validation script creation**
- [x] **Real-time detection verification with human observation**

### Validation Criteria (All Met)

- ✅ Camera capture working (1080p, 10/10 frames)
- ✅ MediaPipe loads without errors
- ✅ Failure modes handled gracefully
- ✅ Code review passed
- ✅ **Visual detection verification complete**
- ✅ **Human-observable validation complete**

**Completion Date**: February 10, 2026  
**Work Log**: [2026-02-10_phase1-quality-correction.md](work-logs/2026-02-10_phase1-quality-correction.md)

---

## Phase 1-2: CLIP Integration (Partial - Requires Realignment)

**Duration**: Weeks 1-2  
**Started**: February 10, 2026  
**Completed**: February 11, 2026 (technical implementation)  
**Research Correction**: February 12, 2026 (scope invalidation)  
**Dependencies**: Phase 1-1 complete ✅ | Dependency stabilization complete ✅  
**Status**: ⚠️ **PARTIAL** - Implementation functional, but vocabulary and evaluation require realignment

### Tasks (5/5)

- [x] Install CLIP dependencies (included in `pip install -e .`)
- [x] Create `src/semantic_hgr/representation/clip_encoder.py`
- [x] Create `src/semantic_hgr/semantics/clip_intent.py`
- [x] Integration test: Camera → MediaPipe → CLIP → embedding
- [x] Validate pipeline latency <100ms

**Completion Date**: February 10, 2026  
**Work Log**: [2026-02-10_clip-integration.md](work-logs/2026-02-10_clip-integration.md)

### Validation Results

**Technical Implementation (VALID)**:

- ✅ Model loading: ViT-B/32 with OpenAI weights
- ✅ Embedding generation: 512-dim, L2-normalized
- ✅ Pipeline integration: Layers 1-3 operational
- ✅ **End-to-end latency: 42.4ms** (well below <100ms target)
- ✅ Phase 1-1 integrity: All Layer 1 components functional

**Semantic Validation (INVALIDATED - Feb 12)**:

- ❌ **19-intent vocabulary**: 68% contain dynamic gestures (out of scope)
- ❌ **Top-1 probability 0.960**: Reflects text embedding similarity only, NOT visual grounding
- ❌ **Synthetic calibration**: Random embeddings lack semantic structure
- ❌ **Confidence scores**: Meaningless without valid ground truth

**Corrected Assessment**:

- ✅ CLIP encoder functional for static hand pose embedding
- ❌ Semantic intent extraction NOT validated for dynamic gestures
- ❌ Probability calibration NOT applicable to current setup
- ⚠️ **REALIGNMENT REQUIRED**: Replace vocabulary with static-only intents

---

## Phase 1-3: Immediate Realignment Action Plan (Static Gesture Track)

**Duration**: Weeks 2-3  
**Dependencies**: Research correction complete ✅  
**Status**: 🔴 **ACTIVE** - Realignment in progress  
**Objective**: Align implementation with corrected research scope (static gestures only)

### Step-by-Step Action Plan

#### Step 1: Vocabulary Audit and Freeze ⏳

- [ ] **Freeze current 19-intent vocabulary** (no modifications until audit complete)
- [ ] **Classify each intent**: Static (single-frame) vs Dynamic (multi-frame) vs Ambiguous
- [ ] **Documentation**: Create `INTENT_CLASSIFICATION.md` with technical justification
- [ ] **Expected Output**: Table mapping each intent to category with frame requirements

**Acceptance Criteria**: All 19 intents labeled as Static/Dynamic/Ambiguous with citations to HaGRID/ActionCLIP papers.

#### Step 2: Static-Only Vocabulary Definition ⏳

- [ ] **Define Tier-1 static vocabulary**: 9 HaGRID-aligned intents (thumbs_up, thumbs_down, open_palm, closed_fist, peace_sign, ok_sign, pointing, call_gesture, no_hand)
- [ ] **Create new constant**: `STATIC_SEMANTIC_INTENTS` in `clip_intent.py`
- [ ] **Map to HaGRID classes**: Document correspondence with dataset taxonomy
- [ ] **Validation**: Verify each intent recognizable from single frame

**Acceptance Criteria**: Static vocabulary contains ONLY pose-based gestures with clear visual grounding.

#### Step 3: Pipeline Configuration Update ⏳

- [ ] **Disable dynamic intents**: Remove scroll*\*, swipe*_, zoom\__, rotate\_\*, drag from active pipelines
- [ ] **Update demo config**: Switch to `STATIC_SEMANTIC_INTENTS` vocabulary
- [ ] **Add runtime validation**: Reject predictions for disabled intents
- [ ] **Configuration flag**: Add `--static-only` mode to all entry points

**Acceptance Criteria**: Demo and evaluation scripts operate exclusively on static vocabulary when flag enabled.

#### Step 4: Metric Replacement 🔴

- [ ] **Remove ECE/MCE reporting**: Archive calibration scripts until prerequisites met
- [ ] **Implement Top-K accuracy**: Fraction of samples with correct intent in top-1/3/5
- [ ] **Implement margin analysis**: Confidence gap between top-1 and top-2 predictions
- [ ] **Update Makefile targets**: `make static-eval` replaces `make calibration`

**Acceptance Criteria**: No calibration metrics reported without explicit validation on real static gesture data.

#### Step 5: Calibration Study Redesign 🔴

- [ ] **Dataset collection**: Download HaGRID subset (9 static classes, 500-1000 images each)
- [ ] **Ground truth validation**: Verify single-frame labels unambiguous
- [ ] **Evaluation protocol**: Top-K accuracy, confusion matrix, margin distribution
- [ ] **Human baseline**: Measure inter-rater agreement (Fleiss' Kappa) for ceiling performance

**Acceptance Criteria**: Calibration study operates ONLY on validated static gesture images with human-verified labels.

#### Step 6: Documentation Update 🔴

- [ ] **Update research question**: "Can CLIP associate static hand poses with semantic intent labels?"
- [ ] **Update architecture doc**: Clarify static-only scope, defer dynamic gestures to Phase 3
- [ ] **Update demo README**: Remove claims of dynamic gesture understanding
- [ ] **Add future roadmap**: Document temporal modeling requirements for dynamic gestures

**Acceptance Criteria**: All documentation reflects corrected scope with no overclaims.

#### Step 7: Demo Reclassification 🔴

- [ ] **Rename demo**: "Real-time Static Hand Pose Semantic Recognition"
- [ ] **Add disclaimer overlay**: "Static poses only - motion intents require temporal modeling (future work)"
- [ ] **Update display labels**: Show pose descriptions, not motion-intent names
- [ ] **Add mode toggle**: Static mode (validated) vs Research mode (experimental)

**Acceptance Criteria**: Demo clearly communicates what it proves (pose recognition) vs what it does NOT prove (motion understanding).

#### Step 8: Regression Prevention 🔴

- [ ] **Add unit tests**: Verify static vocabulary enforcement
- [ ] **CI/CD validation**: Fail if dynamic intents appear in evaluation pipelines
- [ ] **Code review checklist**: Require explicit justification for any temporal modeling claims
- [ ] **Documentation review**: Quarterly audit to prevent scope creep

**Acceptance Criteria**: Automated checks prevent reintroduction of invalid assumptions.

---

## System Architecture

**5-Layer Architecture** (see [SEMANTIC_SYSTEM_ARCHITECTURE.md](SEMANTIC_SYSTEM_ARCHITECTURE.md)):

1. **Perception**: Camera + MediaPipe → 224×224 hand crops
2. **Representation**: CLIP encoder → 512-dim embeddings
3. **Semantic Intent**: Gesture → text descriptions via similarity
4. **Command Resolution**: Intent → system commands
5. **Execution**: OS command invocation

**Completed Layers**:

- 1 (Perception) - ✅ 100% complete (camera + MediaPipe)
- 2 (Representation) - ✅ 100% complete (CLIP ViT-B/32 encoder)

**Partially Complete Layers**:

- 3 (Semantic Intent) - ⚠️ **75% complete** - Implementation functional, vocabulary requires realignment

**Blocked Layers** (Dependent on Realignment):

- 4 (Command Resolution) - ⏸️ DEFERRED until static vocabulary validated
- 5 (Execution) - ⏸️ DEFERRED until Layer 4 complete

## **Next Layer**: 3 (Semantic Intent) - 🔴 Realignment required (see Action Plan)

## Dependencies

**Core Dependencies (Stable)**:

- python=3.10, opencv≥4.8.0, numpy≥1.24.0, pillow≥9.0.0, mediapipe==0.10.21

**Layer 1 (Perception)** - ✅ Installed:

- tqdm (progress bars)

**Layer 2 (Representation)** - ✅ Installed:

- PyTorch 2.10.0, torchvision 0.25.0, open-clip-torch 3.2.0

**Layer 3 (Semantic Intent)** - ✅ Implemented:

- (no additional external dependencies beyond Layer 2)

**Layer 4 (Command Matching)** - ⏳ Pending (Phase 1-3):

- sentence-transformers ≥2.2.0

---

## Recent Milestones

- **2026-02-12**: ✅ **Documentation restructuring complete** - Eliminated redundancy, consolidated 6 reports into 2 work-logs
- **2026-02-12**: ✅ **Production pipeline stabilization complete** - All 7 console entry points functional, Makefile created
- **2026-02-11**: ✅ **Phase 1-6 production finalization complete** - Calibration tools, overlap analysis, real-time demo
- **2026-02-11**: ✅ **MATHEMATICAL CORRECTION APPLIED** - Fixed missing softmax, confidence now valid probabilities [0,1]
- **2026-02-11**: ✅ **Phase 1-2 VALIDATED** - 96.0% top-1 probability, 98% success rate (≥0.80 threshold)
- **2026-02-10**: ✅ **Phase 1-2 COMPLETE** - CLIP integration validated, 75.6ms mean latency
- **2026-02-10**: ✅ **Layer 2 & 3 implemented** - CLIPEncoder + SemanticIntentExtractor
- **2026-02-10**: ✅ **Phase 1-1 complete** - Visual validation with human verification

---

## Current Blockers

**CRITICAL (Must Resolve Immediately)**:

1. 🔴 **Vocabulary Realignment** - 68% of intents require temporal modeling (out of scope)
2. 🔴 **Evaluation Metrics** - ECE/MCE invalid without ground truth; must replace with Top-K accuracy
3. 🔴 **Dataset Collection** - Need HaGRID static gesture subset for valid evaluation
4. 🔴 **Documentation Correction** - Architecture doc contains contradictions (static system with dynamic intents)

**Known Limitations (Design Constraints)**:

- ⚠️ **Single-frame inference only** - CLIP has zero temporal modeling capability
- ⚠️ **Static poses only** - Dynamic gestures require ActionCLIP/VideoCLIP (8-16 frame windows)
- ⚠️ **No motion understanding** - Cannot distinguish swipe_left from swipe_right in freeze-frame
- ⚠️ **Calibration invalid** - Synthetic random embeddings lack semantic structure
- ⚠️ **Layers 4-5 unimplemented** - Command resolution blocked until vocabulary validated

---

## Next Immediate Actions

### PRIORITY 1: Realignment Execution (BLOCKING)

**Week 1 (Immediate)**:

1. [ ] **Vocabulary audit** - Classify all 19 intents as Static/Dynamic/Ambiguous
2. [ ] **Define Tier-1 vocabulary** - 9 HaGRID-aligned static intents
3. [ ] **Update clip_intent.py** - Add `STATIC_SEMANTIC_INTENTS` constant
4. [ ] **Disable dynamic intents** - Remove from active evaluation pipelines

**Week 2 (Metrics Replacement)**:

5. [ ] **Archive calibration scripts** - Move to `scripts/archive/` until prerequisites met
6. [ ] **Implement Top-K accuracy** - New evaluation metric for static gestures
7. [ ] **Implement margin analysis** - Confidence gap between top-1 and top-2
8. [ ] **Update Makefile** - Replace `make calibration` with `make static-eval`

**Week 3 (Dataset Collection)**:

9. [ ] **Download HaGRID subset** - 9 static classes (like, dislike, palm, fist, peace, ok, point, call, no_gesture)
10. [ ] **Prepare evaluation set** - 500-1000 images per class, train/val/test splits
11. [ ] **Run static evaluation** - Top-1/3/5 accuracy on real data
12. [ ] **Human baseline study** - Measure inter-rater agreement for ceiling performance

### PRIORITY 2: Documentation Correction (NON-BLOCKING)

- [ ] Update [SEMANTIC_SYSTEM_ARCHITECTURE.md](../architecture/SEMANTIC_SYSTEM_ARCHITECTURE.md) - Remove dynamic gesture claims
- [ ] Update demo README - Clarify static-only scope
- [ ] Create INTENT_CLASSIFICATION.md - Document static vs dynamic taxonomy
- [ ] Add future roadmap - Temporal modeling requirements

### PRIORITY 3: Future Research Paths (DEFERRED)

**Phase 2: Temporal Modeling (Multi-Frame)** - ⏸️ Future Work

- [ ] Evaluate ActionCLIP integration (requires 8-16 frame windows)
- [ ] Design temporal aggregation architecture (LSTM vs Transformer)
- [ ] Estimate latency trade-offs (8× increase for 8-frame window)
- [ ] Collect or generate video gesture dataset

**Phase 3: Optical Flow Heuristics (Lightweight)** - ⏸️ Future Work

- [ ] Implement Lucas-Kanade optical flow
- [ ] Design pose + motion fusion rules
- [ ] Evaluate motion estimation noise tolerance
- [ ] Compare to full temporal modeling

**Current Status:** Phase 1-2 implementation complete but vocabulary and evaluation invalidated. Realignment action plan active. Layers 1-2 functional, Layer 3 requires vocabulary correction.

## Success Metrics - Phase 1-1 (Complete)

| Metric                   | Target         | Achieved         |
| ------------------------ | -------------- | ---------------- |
| Camera capture           | ✅ Working     | ✅ 1080p, stable |
| Detection confidence     | >50%           | ✅ 99.1% avg     |
| Processing speed         | <100ms         | ✅ 13.7ms avg    |
| Visual verification      | ✅ Manual test | ✅ **COMPLETE**  |
| Detection reliability    | Observable     | ✅ **VERIFIED**  |
| **Dependency stability** | Clean install  | ✅ **VALIDATED** |

## Success Metrics - Phase 1-2 (Partial)

### Technical Implementation (VALIDATED)

| Metric                 | Target       | Achieved              | Status  |
| ---------------------- | ------------ | --------------------- | ------- |
| CLIP installation      | ✅ `.[clip]` | ✅ Complete (3.2.0)   | ✅ PASS |
| PyTorch version        | ≥2.0.0       | ✅ 2.10.0             | ✅ PASS |
| Model loading          | ViT-B/32     | ✅ **OpenAI weights** | ✅ PASS |
| Embedding generation   | 512-dim      | ✅ **L2-normalized**  | ✅ PASS |
| Encoding latency       | <50ms        | ✅ **0.03ms**         | ✅ PASS |
| **End-to-end latency** | <100ms       | ✅ **42.4ms**         | ✅ PASS |
| **Unit tests**         | Passing      | ✅ **22/22 passed**   | ✅ PASS |
| **Phase 1-1 intact**   | Functional   | ✅ **Verified**       | ✅ PASS |

### Semantic Validation (INVALIDATED - Feb 12, 2026)

| Metric                     | Target                    | Achieved             | Status       |
| -------------------------- | ------------------------- | -------------------- | ------------ |
| **Vocabulary Validity**    | Static-only gestures      | ❌ **68% dynamic**   | ❌ FAIL (🔴) |
| **Ground Truth**           | Single-frame recognizable | ❌ **Undefined**     | ❌ FAIL (🔴) |
| **Calibration Data**       | Real gesture images       | ❌ **Synthetic**     | ❌ FAIL (🔴) |
| **Top-1 Probability**      | Visual grounding          | ❌ **Text-only**     | ❌ FAIL (🔴) |
| **Semantic Understanding** | Intent from visual cues   | ❌ **Not validated** | ❌ FAIL (🔴) |

**🔴 = Invalidated by root cause analysis**

### Corrected Success Criteria (Static Gesture Track)

**Phase 1-3 Realignment Targets**:

| Metric                    | Target               | Status |
| ------------------------- | -------------------- | ------ |
| Static vocabulary defined | 9 HaGRID-aligned     | ⏳ TBD |
| Top-1 accuracy (HaGRID)   | ≥60%                 | ⏳ TBD |
| Top-3 accuracy (HaGRID)   | ≥85%                 | ⏳ TBD |
| Inter-class margin        | ≥0.20                | ⏳ TBD |
| Human agreement (Kappa)   | ≥0.80                | ⏳ TBD |
| Dynamic intents           | 0 in active pipeline | ⏳ TBD |

**Calibration Validity Prerequisites** (NOT YET MET):

1. ❌ Static-only vocabulary enforced
2. ❌ Real gesture images collected
3. ❌ Ground truth validated by human annotators
4. ❌ Top-1 accuracy ≥60% on test set
5. ❌ Task validity confirmed (single-frame sufficient)

**When Calibration Becomes Valid**:  
ONLY after all prerequisites met AND temporal modeling implemented for dynamic gestures OR dynamic gestures permanently removed from scope.

**Previous "Confidence Correction" (Feb 11, 2026)** - Status: ⚠️ INCOMPLETE

- Applied logit_scale (100×) + softmax + temperature scaling
- Achieved 96% top-1 probability
- **BUT**: Probability reflects text embedding similarity, NOT visual grounding
- **Conclusion**: Mathematical correctness ≠ semantic validity

---

## Metric Realignment: Why ECE/MCE Are Currently Inappropriate

### Technical Justification

**Expected Calibration Error (ECE)** measures alignment between predicted confidence and empirical accuracy:

```
ECE = Σ |P(correct | confidence ∈ bin) - avg_confidence(bin)| × P(bin)
```

**Prerequisites for Valid Calibration**:

1. ✅ Model outputs probability distribution: `CLIP + softmax` → Yes
2. ❌ **Ground truth labels exist and are unambiguous**: No (for 68% of vocabulary)
3. ❌ **Task is well-defined**: No (motion intents undefined in single frame)
4. ❌ **Evaluation data matches training distribution**: No (synthetic random embeddings)

**Why Current ECE=0.857 Is Meaningless**:

- **Not overconfidence**: Model is confident because all text embeddings are valid from CLIP's perspective
- **Not miscalibration**: Cannot calibrate when ground truth is unknowable
- **Actual cause**: Task ambiguity (asking unanswerable questions)

**Example Demonstrating Invalidity**:

```
Frame: Hand with index finger extended, position (150, 200)

Possible Labels (all equally valid from single frame):
- "point" (if stationary) → 33% probability
- "swipe_left" (if moving left) → 33% probability
- "swipe_right" (if moving right) → 33% probability

Ground Truth: ??? (motion unknown)

Calibration: Cannot compute accuracy when correct label is undefined
```

### Authoritative Metrics (Immediate)

**For Static Gesture Evaluation**:

1. **Top-K Retrieval Accuracy** (Primary):
   - Top-1: `P(argmax_i p_i = y_true)`
   - Top-3: `P(y_true ∈ top 3 predictions)`
   - Top-5: `P(y_true ∈ top 5 predictions)`
   - **Why valid**: Measures retrieval quality without assuming perfect calibration

2. **Inter-Class Margin** (Confidence Indicator):
   - `margin = p_top1 - p_top2`
   - **Interpretation**: High margin (>0.3) → low ambiguity, Low margin (<0.1) → high ambiguity
   - **Usage**: Reject predictions below margin threshold

3. **Confusion Matrix** (Error Analysis):
   - Identifies systematically confused intent pairs
   - Informs vocabulary refinement (merge similar intents)

4. **Human Agreement Study** (Ceiling Performance):
   - Inter-rater reliability (Fleiss' Kappa)
   - Establishes maximum achievable accuracy
   - Validates intent labels are visually grounded

### When Calibration Becomes Valid Again

**Future Conditions** (NOT current system):

1. ✅ **Vocabulary restricted to static gestures** (9 intents, HaGRID-aligned)
2. ✅ **Real gesture images collected** (no synthetic data)
3. ✅ **Ground truth validated by humans** (Kappa ≥ 0.80)
4. ✅ **Reasonable retrieval accuracy** (Top-1 ≥ 60%)
5. ✅ **Task validity confirmed** (single-frame sufficient for all intents)

**Then and only then**: Apply temperature scaling to align confidence with empirical accuracy.

**Alternative Path** (No Calibration Required):

- Use **margin-based thresholding** instead:
  ```python
  if margin > 0.3:  # High confidence
      accept_prediction()
  else:
      reject_as_ambiguous()  # Requires disambiguation
  ```
- Avoids calibration entirely while maintaining reliability

---

## Demo Reclassification

### Current Demo Purpose (CORRECTED)

**Official Title**: "Real-time Static Hand Pose Semantic Recognition Demo"

**What the Demo PROVES**:

1. ✅ Real-time hand detection (MediaPipe at 13.7ms latency)
2. ✅ CLIP-based feature extraction (ViT-B/32 at 0.03ms encoding)
3. ✅ Vision-language embedding alignment (512-dim semantic space)
4. ✅ End-to-end latency <100ms (42.4ms measured)
5. ✅ Top-K intent retrieval (text similarity to visual embeddings)

**What the Demo DOES NOT PROVE**:

1. ❌ Dynamic gesture understanding (requires temporal modeling)
2. ❌ Motion intent inference (swipe, scroll, zoom out of scope)
3. ❌ Calibrated confidence scores (not validated on real data)
4. ❌ Semantic understanding of motion-dependent actions
5. ❌ Production-ready accuracy (no evaluation on real gestures)

**Why This Framing Is Defensible**:

- **Aligned with CLIP capabilities**: Static image understanding (trained on 400M image-text pairs)
- **Matches HaGRID scope**: Static hand pose recognition (18 pose classes)
- **Literature-backed**: Pose-based recognition achievable from single frame (Rautaray & Agrawal, 2015)
- **Honest about limitations**: Explicit about what requires future work

### Required Demo Corrections

**Immediate Changes**:

1. **Rename**: "Real-time Static Hand Pose Semantic Recognition"
2. **Add disclaimer**: Overlay stating "Static poses only - motion intents require temporal modeling (Phase 3)"
3. **Update labels**: Show pose descriptions ("Index Extended", "Open Palm") not motion intents ("scroll_up", "swipe_left")
4. **Mode toggle**: Static mode (9 validated intents) vs Research mode (full 19 experimental)

**Updated Purpose Statement**:

> "Demonstrates real-time static hand pose recognition using CLIP's vision-language alignment. Single-frame inference achieves 42ms latency with MediaPipe hand detection and ViT-B/32 encoding. Motion-dependent gestures (scroll, swipe, zoom) require temporal modeling (future work, Phase 3)."

---

## Future Research Roadmap

### PATH 1: Static-Only System (ACTIVE)

**Timeline**: Weeks 2-4  
**Status**: 🔴 **IN PROGRESS** (Realignment phase)

**Scope**:

- 9 HaGRID-aligned static intents
- Single-frame RGB inference
- Top-K retrieval accuracy evaluation
- Margin-based prediction filtering

**Advantages**:

- ✅ Scientifically defensible (matches CLIP capabilities)
- ✅ Fast iteration (no temporal modeling complexity)
- ✅ Clear evaluation criteria
- ✅ Production-ready latency (42ms)

**Limitations**:

- ❌ No dynamic gestures (scroll, swipe, zoom)
- ❌ Limited semantic richness (9 intents vs 19)

**Expected Performance**:

- Top-1 accuracy: 60-75% (based on ViT-B baseline in HaGRID paper)
- Top-3 accuracy: 85-90%
- Human ceiling: 90-95% (Kappa ≥ 0.80)

---

### PATH 2: Temporal Modeling for Dynamic Gestures (FUTURE)

**Timeline**: Months 4-6  
**Status**: ⏸️ **DEFERRED** (blocked on static system validation)  
**Prerequisites**: Static system achieving Top-1 ≥60%, vocabulary validated

**Architecture**: Multi-frame CLIP + temporal aggregation

**Required Components**:

1. **Temporal Encoder**:
   - Option A: Temporal Shift Module (TSM) from ActionCLIP
   - Option B: LSTM on CLIP frame embeddings (512-dim → 256-dim hidden state)
   - Option C: Temporal Transformer (X-CLIP architecture)

2. **Frame Window**: 8-16 frames (0.27-0.53 seconds at 30 FPS)

3. **Updated Vocabulary**: Add dynamic gestures
   ```python
   DYNAMIC_INTENTS = {
       "scroll_up": ["upward hand motion", "scroll content up"],
       "scroll_down": ["downward hand motion", "scroll content down"],
       "swipe_left": ["leftward swipe gesture", "move left"],
       "swipe_right": ["rightward swipe gesture", "move right"],
       "zoom_in": ["pinch outward", "expand gesture"],
       "zoom_out": ["pinch inward", "contract gesture"],
   }
   ```

**Implementation Pseudocode**:

```python
class TemporalGestureRecognizer:
    def __init__(self):
        self.clip_encoder = CLIPEncoder()  # Frozen
        self.temporal_lstm = LSTM(input=512, hidden=256, layers=2)
        self.classifier = Linear(256, num_dynamic_intents)

    def recognize(self, frame_sequence):  # Shape: (T, H, W, 3)
        # Extract per-frame CLIP embeddings
        embeddings = [self.clip_encoder.encode_image(f) for f in frame_sequence]
        embeddings_seq = torch.stack(embeddings)  # (T, 512)

        # Temporal aggregation
        temporal_features, _ = self.temporal_lstm(embeddings_seq)

        # Classify from final hidden state
        logits = self.classifier(temporal_features[-1])
        return softmax(logits)
```

**Advantages**:

- ✅ Supports full dynamic gesture vocabulary
- ✅ Leverages CLIP's visual understanding (frozen encoder)
- ✅ Research-backed (ActionCLIP achieves 68% accuracy with TSM)

**Challenges**:

- ❌ Requires video dataset (HaGRID is single-frame only)
- ❌ 8× compute cost (8 frames × CLIP encoding)
- ❌ Latency: ~320ms (8 frames × 40ms) - exceeds <100ms target
- ❌ Complex temporal annotation (gesture start/end frames)

**Recommendation**: DEFER until static system validated and latency requirements relaxed.

---

### PATH 3: Hybrid System (Static Pose + Optical Flow)

**Timeline**: Months 3-4  
**Status**: ⏸️ **EXPLORATORY** (lower priority than Path 2)  
**Prerequisites**: Static system functional

**Architecture**: CLIP for pose + OpenCV optical flow for motion

**Concept**:

1. **Static Pose Classification**: CLIP identifies hand shape (point, palm, fist)
2. **Motion Vector Estimation**: Lucas-Kanade optical flow computes velocity
3. **Intent Fusion**: Combine pose + motion to infer dynamic intent

**Fusion Logic**:

```python
def hybrid_recognition(frame_current, frame_previous):
    # Static pose
    pose = clip_encoder.encode_image(frame_current)
    pose_intent, pose_conf = semantic_extractor.extract_intent(pose)

    # Motion vector
    flow = cv2.calcOpticalFlowFarneback(frame_previous, frame_current, ...)
    motion_vector = np.mean(flow, axis=(0,1))  # Average velocity
    motion_magnitude = np.linalg.norm(motion_vector)

    # Fusion rules
    if motion_magnitude < threshold:  # Stationary
        return pose_intent
    elif pose_intent == "pointing":
        if motion_vector[1] < -v_threshold:  # Upward
            return "scroll_up"
        elif motion_vector[1] > v_threshold:  # Downward
            return "scroll_down"
        elif motion_vector[0] < -h_threshold:  # Leftward
            return "swipe_left"
        elif motion_vector[0] > h_threshold:  # Rightward
            return "swipe_right"
    else:
        return pose_intent
```

**Advantages**:

- ✅ Lightweight (no temporal model training)
- ✅ Preserves CLIP's semantic understanding
- ✅ Adds motion awareness without video dataset
- ✅ Low latency overhead (~5ms for optical flow)

**Limitations**:

- ⚠️ Heuristic fusion rules (not learned end-to-end)
- ⚠️ Optical flow noisy (sensitive to lighting, camera motion)
- ⚠️ Requires careful threshold tuning per environment
- ⚠️ Limited to simple translation gestures (not rotation, pinch)

**Feasibility**: MEDIUM - Good compromise for quick iteration without full temporal modeling.

---

## Work Logs

**Location**: [docs/work-logs/](work-logs/)

**Recent**:

1. [2026-02-11_phase1-2-confidence-fix.md](work-logs/2026-02-11_phase1-2-confidence-fix.md) - **CRITICAL FIX**
2. [2026-02-10_clip-integration.md](work-logs/2026-02-10_clip-integration.md)
3. [2026-02-10_dependency-stabilization.md](work-logs/2026-02-10_dependency-stabilization.md)
4. [2026-02-10_phase1-quality-correction.md](work-logs/2026-02-10_phase1-quality-correction.md)
5. [2026-02-10_documentation-consolidation-webcam-validation.md](work-logs/2026-02-10_documentation-consolidation-webcam-validation-consolidation-webcam-validation.md)
6. [2026-02-10_phase1-corrective-enforcement.md](work-logs/2026-02-10_phase1-corrective-enforcement.md)
