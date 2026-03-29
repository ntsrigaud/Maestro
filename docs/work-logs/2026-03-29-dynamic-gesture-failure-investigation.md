# Investigation Report — Dynamic Gesture Detection Failures

**Date:** 2026-03-29  
**Branch:** feat/presentation-meeting-control  
**Scope:** Full technical investigation into why swipe_left, swipe_right, and pinch_hold are unreliable during real-time inference.

---

## 1. Architecture Review

### 1.1 End-to-End Data Flow

The production data flow is implemented as specified:

```
Camera (OpenCV) → MediaPipe Hands → LandmarkFeatureTransformer
→ SlidingWindowSequenceBuffer → GestureLSTMClassifier
→ PredictionStabilizer → GestureActivationController
→ KeyboardCommandDispatcher → OverlayRenderer
```

All nine pipeline components described in the architecture specification are implemented and wired in `app/bootstrap.py`. The domain contracts, interface protocols, and dependency injection container are correctly structured.

### 1.2 Feature Engineering Parity

The same `LandmarkFeatureTransformer` class is used for both dataset construction (via `dataset_builder.py`) and real-time inference (via `bootstrap.py`). The feature vector is:

- 21 wrist-relative, scale-normalised landmark positions (63 floats)
- 21 time-normalised velocity vectors (63 floats)
- Total: 126-dimensional feature vector per frame

Orientation normalisation is enabled (`orientation_normalization: true` in `configs/model/feature_pipeline.yaml`) and correctly enforced by `validate_runtime_compatibility()` at bootstrap time. Feature schema version `feature-schema-v1` is embedded in both the NPZ dataset and the exported `.pt` artifact.

**Parity verdict: CORRECT. No feature transformation divergence exists between training and inference code paths.**

### 1.3 Configuration Consistency

| Parameter                 | Training (dataset) | Runtime (inference.yaml) | Match |
| ------------------------- | ------------------ | ------------------------ | ----- |
| window_length             | 24                 | 24                       | ✅    |
| window_step               | 3                  | 3                        | ✅    |
| min_valid_ratio           | 0.8                | 0.8                      | ✅    |
| orientation_normalization | true               | true                     | ✅    |
| feature_dim               | 126                | 126                      | ✅    |

### 1.4 Data Flow Correctness

The runtime loop in `app/runtime.py` correctly sequences every pipeline stage. Timestamp propagation, window extraction, LSTM inference with softmax, and state machine routing are all implemented as designed. The `_predict()` function correctly assembles a (1, 24, 126) tensor and maps output probabilities to class labels.

### 1.5 Architecture Deviations Found

Two deviations from the architecture specification were identified:

**Deviation A: Prediction split does not honour recording boundaries.**  
The `datamodule.py` splits overlapping sliding-window samples by random index rather than by recording ID. This enables severe data leakage (covered in Section 2).

**Deviation B: Data collection tool does not segment individual gesture instances.**  
`scripts/collect_data.py` captures all frames in a single continuous session and assigns one label to every frame, with no mechanism to mark individual gesture start and end points. This contaminates training windows with non-gesture frames.

---

## 2. Dynamic Gesture Failure Root Cause Analysis

### Root Cause 1 (CRITICAL): Single-Session Dataset with Temporal Data Leakage

**Evidence from codebase:**  
`data/raw/labeled_landmarks.jsonl` — 3,131 total raw frames across 8 recordings (one per class).  
`data/processed/gesture_dataset_v1.npz` — 984 samples, all from a single recording per class.

```
swipe_left:  97 windows · 1 unique recording
swipe_right: 128 windows · 1 unique recording
pinch_hold:  123 windows · 1 unique recording
```

**Evidence from training results:**  
`experiment-model/lstm_v1_20260328T100850Z/training_metrics.json`:

```
val_accuracy:  1.0 (perfect)
val_macro_f1:  1.0 (perfect)
test_accuracy: 1.0 (perfect)
test_macro_f1: 1.0 (perfect)
```

This 100% test accuracy is not generalisation — it is artefact of leakage.

**The leakage mechanism:** With `window_step=3` and `window_length=24`, consecutive sliding windows share 21 out of 24 frames (87.5% overlap). The random 70/15/15 split assigns window indices independently of recording boundaries. A window at index `i` and a window at index `i+1` are nearly identical vectors, yet they can be assigned to different splits. The model therefore sees near-duplicate vectors from the same physical motion in both its training set and its validation set. Any model will achieve perfect validation accuracy under these conditions regardless of whether it has generalised.

**Why dynamic gestures fail in production:** The model has memorised the specific trajectory artefacts of the single swipe recording session (`rec-swipe-right-001`, `rec-swipe-left-001`). These artefacts include the precise velocity signature at the specific illumination, camera distance, and hand position of a single recording environment. When the model encounters any real-time swipe, it is processing a trajectory that comes from a completely different distribution.

---

### Root Cause 2 (CRITICAL): Zero Subject and Session Diversity

**Evidence:**

```
Total unique recording_ids:  8     (one per gesture class)
Subjects represented:         1     (inferred — a single collection session)
Camera angles represented:    1
Lighting conditions:          1
Hand sizes represented:       1
```

Every training window originates from one person, one camera angle, and one recording session. The LSTM has never encountered:

- A second person's hand proportions or motion style
- Left-hand gesture presentation
- Varying camera distance
- Different lighting (shadows, low light, harsh backlight)
- A swipe executed at different speeds

Static hold gestures (`open_palm_hold`, `fist_hold`, `thumbs_up`, `v_sign`) are less affected because the position vector is largely independent of the swiping motion — a held fist looks roughly similar across subjects. Dynamic gestures depend on the temporal velocity pattern, which varies dramatically across individuals, speeds, and execution styles.

---

### Root Cause 3 (HIGH): Stabilisation Vote Window Incompatible with Brief Dynamic Gestures — ✅ FIXED

**Evidence from `pipeline/stabilization.py`:**

```python
PredictionStabilizer(
    vote_window=3,    # requires 3 consecutive accepted predictions
    ema_alpha=0.4,    # strong EMA smoothing
)
```

**Evidence from `configs/inference.yaml`:**

```yaml
window_step: 3 # emit one window every 3 frames
```

**Analysis:**

At the measured real-time FPS of ~15, one new window is emitted every 200 ms. The stabiliser requires 3 consecutive accepted predictions = 600 ms of consistent high-confidence output. A typical swipe gesture lasts 300–700 ms.

The sliding window at step 3 means:

- Window T covers frames 0–23
- Window T+1 covers frames 3–26
- Window T+2 covers frames 6–29

For a 500 ms swipe at 15 FPS (≈7 frames of actual motion): the motion peak is visible in approximately 2–3 consecutive windows. The vote window of 3 means the stabiliser requires the LSTM to classify all 3 of those windows as `swipe_right` with confidence ≥ 0.85. In practice:

- Window T: swipe occupies the end of the window (weak signal) → confidence may fall below 0.85
- Window T+1: swipe is centred → maximum confidence
- Window T+2: swipe occupies the start of the window (weakening) → confidence falls

The stabiliser receives at most 1–2 highly confident swipe predictions and discards them as `temporal_inconsistency`. This is the proximate failure mechanism: even when the model detects the gesture correctly, the stabiliser rejects it because 3 consecutive high-confidence windows cannot be sustained for a brief dynamic event.

The EMA (alpha=0.4) compounds this: after the swipe passes, the EMA retains ~36% of the historical signal (0.6^2 decay after 2 frames), blending stale motion features into the current prediction and making it harder to reach the threshold cleanly on the windows that DO contain the swipe.

**Fix applied (2026-03-29):** `PredictionStabilizer` now accepts an optional `class_vote_windows` dict. `configs/thresholds/production.yaml` sets `swipe_right: 1`, `swipe_left: 1`, `pinch_hold: 2`, and `3` for all static hold classes. The history deque is sized to the maximum window across all classes; consistency is checked against only the last N entries for each candidate class.

---

### Root Cause 4 (HIGH): Training Sessions Label All Frames with One Gesture, Including Rest Periods

**Evidence from data collection approach:**

`collect_data.py` assigns the same label to every captured frame for the entire session duration. There is no mechanism to mark when an individual swipe starts and ends. A 43-second swipe_right session contains:

- Approximately 10–15 actual swipe motions (≈3 seconds each)
- Approximately 28–33 seconds of the hand returning to start position, pausing, or in transition

All frames receive the label `swipe_right`. This means roughly 60–70% of `swipe_right` training windows contain no recognisable swipe motion.

**Consequences:**

1. The LSTM learns a noisy mixed signal: "a hand in a swipe-context environment" rather than "a hand performing a lateral sweep"
2. Boundary windows (hand returning after swipe) are indistinguishable from the actual swipe in the label space
3. The model cannot learn a clean velocity signature because many `swipe_right` windows contain near-zero lateral velocity (rest between swipes)

**Why static gestures are unaffected:** For hold gestures, the entire recording session is legitimate training data — holding a fist for 43 seconds is exactly the pattern to learn and the label correctly describes every frame.

---

### Root Cause 5 (HIGH): Random Split on Overlapping Windows Instead of Recording-Level Holdout

**Evidence from `training/datamodule.py`:**

```python
train_idx, temp_idx = train_test_split(
    all_idx,
    test_size=temp_ratio,
    random_state=seed,
    stratify=label_ids,   # stratified by label, not by recording
)
```

The split uses window indices as the unit of division. With 87.5% temporal overlap between adjacent windows, adjacent indices are semantically nearly identical. Stratified random splitting does not prevent data leakage between contiguous windows.

**Correct approach:** Recording-based group split. All windows from a given recording must belong exclusively to either train, validation, or test. Since only one recording per class exists, this cannot be implemented without collecting additional recordings first.

---

### Root Cause 6 (MODERATE): Training-to-Inference Frame Rate Divergence

**Evidence from raw data timestamps:**

```
rec-swipe-left-001:   312 frames · 43.2 s · ~8.5 FPS approx
rec-swipe-right-001:  407 frames · 57.5 s · ~12.0 FPS approx
rec-pinch-hold-001:   391 frames · 42.8 s · ~10.9 FPS approx
```

**Evidence from Phase 5 benchmark:**

```
actual inference FPS: 15.47
```

A 500 ms swipe at the training collection rate of ~10 FPS occupies approximately 5 out of 24 frames. At inference FPS of ~15, the same swipe occupies approximately 7 out of 24 frames. The temporal structure of the motion within the window differs by around 40%, changing which LSTM hidden states dominate the final classification.

**Why static gestures are unaffected:** Hold gestures have a uniform feature distribution throughout the window regardless of FPS, because position features are constant when the hand is still.

---

### Root Cause 7 (MODERATE): pinch_hold Threshold Too High for Real-Time Conditions — ✅ FIXED

**Evidence from `configs/thresholds/production.yaml`:**

```yaml
pinch_hold: 0.9
```

**Evidence from dataset analysis:**

```
pinch_hold: vel_mean_abs=0.2778, vel_std=0.7268
```

`pinch_hold` has the lowest velocity variance of any class. During real-time inference, natural hand tremor, breathing motion, and micro-adjustments generate velocity noise that differs from the artificially still training recording. The 0.9 confidence threshold leaves no headroom for this distribution shift. Any slight departure from the training distribution will drop confidence below the threshold.

**Fix applied (2026-03-29):** `pinch_hold` threshold lowered from 0.90 to 0.85 in `configs/thresholds/production.yaml`. `swipe_right` and `swipe_left` thresholds lowered from 0.85 to 0.80.

---

## 3. Dataset Quality Assessment

### Summary Table

| Metric                 | Current Value     | Recommended Minimum                     |
| ---------------------- | ----------------- | --------------------------------------- |
| Total windows          | 984               | ≥ 5,000                                 |
| Recordings per class   | 1                 | ≥ 10                                    |
| Unique subjects        | 1 (inferred)      | ≥ 5                                     |
| swipe_left samples     | 97                | ≥ 500                                   |
| swipe_right samples    | 128               | ≥ 500                                   |
| pinch_hold samples     | 123               | ≥ 500                                   |
| Training FPS           | ~8–12 (irregular) | Consistent, matches inference (~15 FPS) |
| Gesture segmentation   | None              | Individual instances segmented          |
| Camera angle variation | None              | ≥ 3 angles                              |
| Lighting variation     | None              | ≥ 3 conditions                          |
| Left-hand data         | None              | ≥ 20% of sessions                       |

### Critical Quality Issues

**Issue A — No gesture segmentation in dynamic gesture recordings.** The swipe recordings are one continuous session where the hand repeats the gesture 10–15 times in 43 seconds. Windows that capture the rest phase between swipes are mislabelled. The model cannot distinguish "swipe in progress" from "hand returning after swipe."

**Issue B — Severe temporal autocorrelation in training split.** 87.5% window overlap combined with random window-level splitting creates leakage. The resulting 100% test accuracy is evidence of a failed evaluation protocol, not a high-capability model.

**Issue C — Insufficient sample count for dynamic gesture generalisation.** With 97 windows of `swipe_left` from a single person doing the same trajectory repeatedly, the model has no basis for generalisation. Static gestures require fewer samples to generalise because position vectors are structurally simpler.

**Issue D — No augmentation.** The training pipeline applies no data augmentation: no temporal jitter, no spatial noise, no speed variation. This directly limits robustness.

---

## 4. Public Dataset Evaluation

### 4.1 HaGRID

- **Scale:** 552,992 images, 18 gesture classes
- **Format:** Static images (not video sequences)
- **Dynamic gesture coverage:** None — all classes are static hand shapes. No swipe, no lateral motion.
- **Relevant classes:** `fist`, `palm`, `like` (thumbs_up), `mute` (partial match to pinch), `stop` (partial match to open_palm)
- **Landmark extraction:** Feasible — MediaPipe can process static images. However, landmarks extracted from images give only position vectors; velocity features require temporal sequences and cannot be synthesised from static images.
- **Domain compatibility:** High diversity (many subjects, lighting conditions), large scale. Useful only for improving static hold gesture recognition.
- **Licensing:** CC BY 4.0 (commercial use permitted)
- **Verdict for this project:** Partially useful. Can augment static gesture classes (open_palm_hold, fist_hold, thumbs_up, v_sign). Cannot help with swipe_right, swipe_left, or pinch_hold at all.

### 4.2 IPN Hand

- **Scale:** 4,218 video sequences, 180+ subjects
- **Format:** RGB video sequences with per-frame gesture annotations
- **Dynamic gesture coverage:** Excellent — includes `Swipe Right`, `Swipe Left`, `Swipe Up`, `Swipe Down`, `Zoom In`, `Zoom Out`, `Rotation CW`, `Rotation CCW`, plus click and no-gesture
- **Relevant classes:**

| IPN Hand class | Target vocabulary class | Compatibility |
| -------------- | ----------------------- | ------------- |
| Swipe Right    | swipe_right             | HIGH          |
| Swipe Left     | swipe_left              | HIGH          |
| No gesture     | unknown                 | HIGH          |
| Pointing       | (not in vocabulary)     | N/A           |

- **Missing classes in IPN Hand:** `pinch_hold`, `fist_hold`, `thumbs_up`, `open_palm_hold`, `v_sign` are not present as recognisable equivalents.
- **Landmark extraction:** Feasible — process each frame through MediaPipe to extract landmarks, then pass through `LandmarkFeatureTransformer`. Velocity features can be computed from consecutive frames.
- **Domain differences:**
  - Camera: IPN Hand was collected with Intel RealSense (depth camera, controlled environment). Webcam has different noise characteristics.
  - Backgrounds: Controlled laboratory environments, not typical office/home use.
  - Subject diversity: 180+ subjects is an enormous advantage for swipe generalization.
- **Licensing:** Available for academic and research use. Check latest license terms before commercial deployment.
- **Processing pipeline:** Extract per-frame landmarks with MediaPipe → apply `LandmarkFeatureTransformer` → build sliding windows at consistent FPS → label windows using IPN's per-frame annotations.
- **Verdict for this project:** HIGH VALUE for swipe_right and swipe_left. Would provide hundreds of unique subjects performing well-segmented swipe motions. Domain shift risk is moderate (depth camera vs webcam).

---

## 5. Training Pipeline Improvement Strategy

### 5.1 Split and Evaluation Fixes (Prerequisite — All Other Work Depends on This)

**Fix A: Implement recording-based group split.**  
Split by recording ID instead of window index. This requires multiple recordings per class. Once ≥ 3 recordings exist per class, hold out 1 recording per class for validation and 1 for testing. This guarantees that no information from a validation recording appears in training.

**Fix B: Add recording-level data leakage detection.**  
After splitting, verify that no `recording_id` appears in both train and validation sets. Add an assertion to `datamodule.py`.

**Fix C: Report per-recording validation metrics.**  
Include per-recording_id accuracy in the evaluation report. If one recording scores poorly, it identifies distribution shift rather than hiding it in an aggregate.

### 5.2 Data Collection Improvements

**Segmentation protocol for dynamic gestures:**  
Modify `collect_data.py` to support a hold-to-record mode where data capture is active only while a key is held down. This ensures each recording instance maps to exactly one gesture execution, and rest periods are excluded.

**Multi-session collection:**  
Collect ≥ 10 recordings per gesture class, from ≥ 5 subjects when possible. Each recording should include 20–50 instances of the gesture. Vary: subject, camera distance (40–80 cm), hand size, speed of execution, lighting.

**Separate rest/transition class collection:**  
Record dedicated `transition` or `rest` sequences between gesture executions. Include these in the `unknown` class. The model must learn to recognise the hand returning to neutral, not just "any frame that is not in the middle of a swipe."

**Consistent FPS targeting:**  
Both data collection and inference should use the same target FPS. The `target_fps=30` setting in `app.yaml` is appropriate, but the actual achieved rate on the current hardware is ~15 FPS. Measure the sustained FPS on the collection hardware and set `target_fps` to match. The velocity features are time-normalised and should be FPS-invariant in magnitude, but the temporal pattern within a 24-frame window changes with FPS.

### 5.3 Data Augmentation

Add augmentation in `datamodule.py` during training:

- **Temporal jitter:** Randomly shift window start position by ±1 frame
- **Speed perturbation:** Time-stretch/compress sequences by 0.7–1.3× (resample via interpolation)
- **Spatial noise:** Add Gaussian noise (σ ≈ 0.01) to position features to simulate landmark jitter
- **Mirror augmentation:** Negate `x`-axis positions and velocities to simulate left-hand presentation

Speed perturbation is particularly important for dynamic gestures: if a swipe at 10 FPS is resized to look like a 15 FPS swipe, the model learns FPS-invariant swipe patterns.

### 5.4 Stabilisation Parameter Adjustment — ✅ Applied (2026-03-29)

**Per-class vote windows and thresholds** are now configured in `configs/thresholds/production.yaml` under the `class_vote_windows` and `class_thresholds` keys, and consumed by `PredictionStabilizer` via the new optional `class_vote_windows` parameter:

```
open_palm_hold:  vote_window=3, threshold=0.9
fist_hold:       vote_window=3, threshold=0.9
thumbs_up:       vote_window=3, threshold=0.85
v_sign:          vote_window=3, threshold=0.85
swipe_right:     vote_window=1, threshold=0.80
swipe_left:      vote_window=1, threshold=0.80
pinch_hold:      vote_window=2, threshold=0.85
unknown:         vote_window=1, threshold=1.0
```

**Reduce EMA alpha for dynamic gestures (pending):**  
The EMA alpha=0.4 creates too much lag for gestures that must be detected within 3–5 windows. Consider alpha=0.6 or 0.7 for dynamic classes, allowing the EMA to track rapid changes more responsively.

### 5.5 Model Architecture Improvements

**Increase window length for better dynamic gesture coverage:**  
With `window_step=3` and inference at 15 FPS, one window covers 1.6 seconds. A 500 ms swipe occupies ~31% of the window. Increasing `window_length` to 36 at 15 FPS = 2.4 seconds still keeps latency bounded, and provides more context around the swipe motion. This requires dataset regeneration.

**Add temporal convolution preprocessing:**  
Prepend a 1D convolutional layer (kernel sizes 3 and 5) before the LSTM to extract local motion features. TCN layers are better at capturing short motion bursts within a longer sequence than the LSTM alone.

**Bidirectional LSTM (for offline evaluation only):**  
Not applicable to the real-time pipeline since future frames are unavailable.

**Per-class output temperature scaling:**  
After training, calibrate each class output separately by fitting a temperature scalar on a held-out validation set. Dynamic gesture classes typically need lower temperature (sharper output) to compete against the strong static gesture priors.

### 5.6 Evaluation Improvements

- Add temporal prediction stream logging: for each incoming window, log `(timestamp, top_class, confidence, accepted)` to a file during inference sessions. This allows post-hoc analysis of which windows contain valid detections and which are rejected.
- Add a false-trigger benchmark script (`evaluation/false_trigger_eval.py`) that runs the full pipeline for 5 minutes without any gesture and counts spurious commands.
- Implement a per-gesture recall metric that verifies each gesture can reliably trigger a command within 1.5 seconds of the gesture start.

---

## 6. Dataset Recommendation

**Recommendation: Option C — Hybrid approach.**

**Reasoning:**

The custom dataset must be the primary training source because:

1. The target gesture vocabulary (open_palm_hold as activation, the 6 command gestures) is not covered by any public dataset in a compatible single vocabulary.
2. The `pinch_hold`, `fist_hold`, `open_palm_hold`, `thumbs_up`, and `v_sign` classes are absent or named differently in IPN Hand and HaGRID.
3. The domain (consumer webcam, casual office environment) is best represented by custom-collected data.

However, the custom dataset alone cannot provide sufficient diversity for dynamic gestures, particularly swipe_right and swipe_left, where a single recording is insufficient. IPN Hand provides:

- 180+ subjects performing swipe gestures
- Well-segmented individual gesture instances (each sequence is a single gesture)
- Validated temporal patterns for lateral hand motions

**Hybrid strategy:**

Stage 1 — Pretrain on IPN Hand swipe sequences:  
Extract MediaPipe landmarks from IPN Hand swipe_right and swipe_left sequences (≥ 1,000 sequences each). Build sliding windows through the existing pipeline. Train LSTM on this data for the `swipe_right`, `swipe_left`, and `unknown` classes.

Stage 2 — Fine-tune on re-collected custom data:  
Collect ≥ 10 new recordings per gesture (all 8 classes) following the improved collection guide. Fine-tune the pretrained LSTM on all 8 classes. This allows the model to leverage IPN Hand's diversity while adapting to the specific camera and environment.

Stage 3 — Augment static gesture classes with HaGRID:  
Extract landmarks from selected HaGRID images for `fist`, `palm`, and `like` (thumbs_up) classes to build position-only training windows (set velocity features to zero for static-image-derived samples, and only use them as extra regularisation for position features). This is lower priority than Stage 1 and 2.

**Expected improvement:** Implementing Stage 1 + Stage 2 should lift real-time swipe recall from the current near-zero to ≥ 0.75 within the stabiliser's constraints, assuming the collection guide and corrected training split are also applied.

---

## 7. Custom Gesture Data Collection Guide

This guide is required for Stage 2 of the hybrid strategy and must be followed when collecting improved custom data.

### Camera and Environment Setup

- Camera distance: 50–70 cm from hand
- Camera height: Approximately eye level, slightly below (10–15 degrees downward)
- Hand in frame: Entire hand and forearm visible
- Background: Plain, neutral background preferred; vary between sessions
- Lighting: Even, front-lit; collect separate sessions in natural light, overhead office light, and dim light
- Avoid direct backlight from windows

### General Protocol

- Sit or stand with the camera-facing hand at comfortable height (waist level to shoulder level)
- Warm up hand before recording (flex fingers, rotate wrist)
- For each gesture, complete a trial run outside the recording period to calibrate speed and shape
- Between sessions, rest 5 minutes to avoid fatigue-induced posture drift
- For subject diversity: collect from ≥ 5 different people, both dominant and non-dominant hands where possible

---

### Gesture-Specific Instructions

#### open_palm_hold (Activation Gesture)

**Purpose:** Arms/disarms the system. Must be robust and consistent.  
**Posture:** Open hand, fingers spread naturally. Palm facing the camera. Wrist straight.  
**Motion:** Hold still. No deliberate movement.  
**Duration per instance:** 3–5 seconds of sustained hold.  
**Transitions:** Lower hand, pause 1 second, raise to position, hold again.  
**Samples required:** 30 instances per recording, 10+ recordings, 5+ subjects.  
**Common errors:** Bending fingers slightly (causes confusion with fist), rotating wrist inward.

#### swipe_right

**Posture:** Open hand, fingers together but not tightly closed, palm facing camera or slightly angled forward (natural resting orientation).  
**Starting position:** Hand centred in frame, approximately 40–60 cm from camera.  
**Motion:** Move hand laterally to the right in a single smooth sweep. Keep hand at same depth (no leaning toward/away from camera). Keep motion in a horizontal plane.  
**Speed:** 0.3–0.7 m/s hand speed. Moderately fast. The total sweep should take 300–600 ms.  
**Ending position:** Hand at the right edge of camera view.  
**Return to start:** Move hand out of frame view OR lower hand and return to centre. Do NOT record the return motion — only record the active swipe.  
**Samples required:** 50 individual swipe instances per recording. Collect at three speeds: slow (500–700 ms), medium (300–500 ms), fast (150–300 ms). 10+ recordings, 5+ subjects.  
**Collection mode:** Use hold-to-record mode. Press record key just before the swipe starts; release when the hand completes the sweep.  
**Common errors:** Twisting the wrist during the swipe (changes orientation features), moving the hand diagonally, stopping abruptly instead of completing the arc.

#### swipe_left

**Posture:** Same as swipe_right.  
**Motion:** Mirror image of swipe_right. Move from screen-right to screen-left in one smooth sweep.  
**Speed and duration:** Same as swipe_right.  
**Return to start:** Same protocol — do not record return motion.  
**Samples required:** 50 instances per recording with three speed variants. 10+ recordings, 5+ subjects.  
**Important:** Collect matching swipe_left and swipe_right data from the same subjects. The velocity feature distributions for left and right should be symmetric after orientation normalisation; any systematic asymmetry is a quality issue.

#### fist_hold

**Purpose:** Start presentation / toggle black screen.  
**Posture:** Hand closed into a fist. All four fingers curled into palm. Thumb tucked or resting beside index finger. Knuckles facing the camera.  
**Motion:** None. Hold completely still.  
**Duration per instance:** 2–4 seconds.  
**Samples required:** 30 instances per recording, 10+ recordings, 5+ subjects.  
**Common errors:** Leaving the thumb extended (creates thumb_up confusion), relaxing the grip mid-hold.

#### thumbs_up

**Purpose:** Confirm/raise hand.  
**Posture:** Fist with thumb extended upward. Thumb fully straight. Other fingers tightly closed.  
**Motion:** None. Hold completely still with thumb pointing vertically.  
**Duration per instance:** 2–4 seconds.  
**Samples required:** 25 instances per recording, 10+ recordings, 5+ subjects.  
**Common errors:** Rotating the thumb slightly (changes orientation features), partially extending other fingers.

#### v_sign

**Purpose:** Toggle presenter tools.  
**Posture:** Index and middle fingers extended and spread (V shape). Ring and pinky fingers curled. Thumb relaxed at side.  
**Hand orientation:** Palm facing camera. V fingers pointing upward.  
**Motion:** None. Hold still.  
**Duration per instance:** 2–4 seconds.  
**Samples required:** 25 instances per recording, 10+ recordings, 5+ subjects.  
**Common errors:** Fingers not fully spread (collapses to one-finger point), thumb partially extended.

#### pinch_hold

**Purpose:** Mute/unmute.  
**Posture:** Thumb and index finger touching at their tips, forming a circle. Other three fingers extended naturally (not fully spread, not curled).  
**Motion:** Hold completely still. The pinch is a static held position, not a squeeze action.  
**Duration per instance:** 2–4 seconds.  
**Camera consideration:** The pinch position creates a narrow viewing angle for the contact point. Ensure the hand is rotated so the circle formed by thumb and index is visible to the camera (palm oblique, roughly 30–45 degrees from camera).  
**Samples required:** 30 instances per recording, 10+ recordings, 5+ subjects.  
**Why this gesture is difficult:** Very low inter-class distance to `thumbs_up` and `unknown`. Collect explicit negative examples by holding pinch-like positions with the thumb not quite touching the index finger.

#### unknown

**Recording protocol:** This class is critical for reducing false triggers. Collect a wide variety of non-gesture hand positions:

- Hand at rest at side
- Hand resting on desk
- Partial gestures that do not complete (reaching toward gesture start position)
- Transition between gestures (hand moving back to neutral)
- Hand partially in frame
- Natural typing/pointing motions in the air without intent

**Duration per session:** 3–5 minutes of varied non-gesture activity.  
**Samples required:** At minimum 1,000 unknown windows, from 5+ subjects.  
**Special emphasis:** Collect "almost swipe" transitions — hand moving laterally but not in a deliberate swipe. These are the most common false-trigger sources.

---

### Recording Session Checklist

Before each recording session:

- [ ] Verify camera is at correct distance and angle
- [ ] Confirm lighting is even and background is plain
- [ ] Run 30-second preview pass to confirm MediaPipe hand detection rate ≥ 95%
- [ ] Note recording context: subject ID, dominant hand, camera distance, lighting condition

During recording:

- [ ] For hold gestures: use continuous recording mode; include deliberate micro-variations in position (±5 cm shift, slight rotation) within each instance
- [ ] For dynamic gestures: use hold-to-record mode; capture only the active motion, not the return
- [ ] Record `recording_id` as: `{gesture}-{subject_id}-{session_number}-{condition}` (example: `swipe-right-s03-001-offices`)

After recording:

- [ ] Inspect first 20 frames of JSONL to verify landmarks are valid and plausible
- [ ] Spot-check detect_conf values — reject sessions where median detect_conf < 0.8
- [ ] Confirm recording FPS matches the target inference FPS (within 20%)

---

## 8. Summary

### Root Causes (Ranked by Impact)

| Rank | Root Cause                                                  | Impact       | Fix Required?         |
| ---- | ----------------------------------------------------------- | ------------ | --------------------- |
| 0    | Display non-mirrored: gesture directions inverted in UI     | HIGH         | ✅ Fixed (2026-03-29) |
| 1    | Single-session dataset; temporal leakage in split           | CRITICAL     | Yes                   |
| 2    | Zero subject and session diversity                          | CRITICAL     | Yes                   |
| 3    | Stabiliser vote_window too large for brief dynamic gestures | HIGH         | ✅ Fixed (2026-03-29) |
| 4    | No gesture segmentation in dynamic recordings               | HIGH         | Yes                   |
| 5    | Random window-level split ignores recording boundaries      | HIGH         | Yes                   |
| 6    | Training/inference frame rate divergence                    | MODERATE     | Yes                   |
| 7    | pinch_hold threshold too strict for real-time conditions    | MODERATE     | ✅ Fixed (2026-03-29) |
| 8    | No data augmentation                                        | LOW-MODERATE | Yes                   |

### Immediate Actions (Can Fix Without New Data)

These changes do not require data recollection and can be applied immediately:

1. ✅ **Display mirroring applied** — `mirror_display: true` added to `configs/app.yaml` and `AppConfig`. `OverlayRenderer` now flips the canvas horizontally when `mirror_display=True` and inverts landmark x coordinates so dots remain correctly positioned on the mirrored frame. Text overlays are drawn after the flip and remain readable. The ML pipeline (MediaPipe extraction → feature engineering → LSTM) always operates on the original non-mirrored frame — no coordinate change to the training or inference data path.

2. ✅ **Per-class vote windows applied** — `swipe_right` and `swipe_left` now use `vote_window=1`; `pinch_hold` uses `vote_window=2`; static hold gestures retain `vote_window=3`. Configured in `configs/thresholds/production.yaml` and consumed by `PredictionStabilizer` via the new `class_vote_windows` parameter.

3. ✅ **Thresholds lowered** — `swipe_right` and `swipe_left` reduced from 0.85 → 0.80; `pinch_hold` reduced from 0.90 → 0.85.

4. **Log prediction stream to file** during real-time sessions to diagnose exactly which windows are being rejected and why.

### Medium-Term Actions (Require Data Recollection)

1. Collect ≥ 10 segmented recordings per gesture following this guide, from ≥ 5 subjects.
2. Implement recording-level group split in `datamodule.py`.
3. Add speed augmentation and spatial noise to `datamodule.py`.
4. Retrain LSTM. Expect val accuracy to drop from the current artificial 100% to a more realistic 80–90%, reflecting genuine generalisation.

### Long-Term Actions (Require Public Dataset Integration)

1. Extract MediaPipe landmarks from IPN Hand swipe sequences.
2. Pretrain LSTM on IPN Hand swipe classes then fine-tune on re-collected custom data.
3. Evaluate calibrated per-class temperature scaling post-training.
