# Model Improvement Plan

**Status as of 2026-04-20** · LSTM v1 · 79.1 % test accuracy (IPN) · root causes identified

This document tracks all planned improvements to the recognition pipeline, organized into four tiers by effort and expected impact. Tier 1 items require no retraining; later tiers add data, features, and architecture changes progressively.

---

## Background

The system achieves 79 % accuracy on held-out IPN Hand test recordings, but swipe gestures fail frequently in real-world webcam use. Five root causes have been confirmed:

| #   | Cause                                                                             | Impact      |
| --- | --------------------------------------------------------------------------------- | ----------- |
| 1   | EMA ramp-up penalty (500 ms to cross threshold for a 300 ms gesture)              | Critical    |
| 2   | Domain gap: IPN lab recordings vs. real webcam environment                        | Critical    |
| 3   | Window contamination: `majority` label strategy mislabels mixed-phase windows     | Significant |
| 4   | Directional signal dilution: 6 / 147 feature dims carry swipe information         | Moderate    |
| 5   | Optimiser plateau at epoch 3 (data diversity ceiling, not a hyperparameter issue) | Moderate    |

---

## Tier 1 — Post-Processing Fixes (No Retraining)

> **Status: applied 2026-04-20**

### T1-A — EMA Bypass for Instant-Dispatch Gestures ✅

**Problem.** `PredictionStabilizer` applied EMA (α = 0.4) to all class probabilities before the threshold gate. Starting from rest state (unknown ≈ 1.0 in EMA), a perfect 0.95 swipe prediction is attenuated to 0.38 on the first frame and takes ~5 consecutive windows (~500 ms) to ramp above the 0.80 threshold. A natural swipe lasts 200–400 ms and is already finished before the gate opens.

**Fix.** Derive `_ema_bypass_labels` from the classes whose `class_vote_windows` entry equals 1 (`swipe_right`, `swipe_left`, `swipe_up`, `swipe_down`). In `_smooth_probs`, those labels return the raw model probability while the internal EMA state continues to advance — so held and zoom gestures remain fully smoothed.

**Files changed.**

- [`src/gesture_control/pipeline/stabilization.py`](../src/gesture_control/pipeline/stabilization.py)
- [`tests/unit/test_prediction_stabilization.py`](../tests/unit/test_prediction_stabilization.py) — two new tests

### T1-B — Diagnostic Threshold Reduction ✅

**Purpose.** Lowering swipe thresholds from 0.80 → 0.60 lets us confirm whether the model is producing sufficient raw confidence for swipes in the real-webcam environment. If swipes now work reliably, the model is healthy and the threshold can be raised again to taste (suggested target: 0.70–0.75). If swipes still fail at 0.60, the domain gap (T2-A) is the binding constraint.

**Files changed.**

- [`configs/thresholds/production_ipn.yaml`](../configs/thresholds/production_ipn.yaml) — swipe thresholds set to 0.60, comment added

**Revert condition.** Once real-webcam performance is verified, raise thresholds to 0.70–0.75 and remove the `DIAGNOSTIC` comment.

---

## Tier 2 — Data Pipeline Improvements (No Architecture Change)

> **Prerequisite**: Tier 1 applied and validated on real webcam.

### T2-A — User-Specific Calibration and Fine-Tuning

**Problem.** The 15 pp train/test accuracy gap is a domain gap. The model memorizes IPN recording characteristics (camera distance ~1–2 m, deliberate slow gestures, controlled lighting). It has never seen the user's hand anatomy, webcam FOV, or gesture speed.

**Plan.**

1. Use the existing `collect_data.py` script to record 3–5 minutes of real gestures — approximately 30–50 repetitions per class at natural speed.
2. Build a small calibration NPZ from the new recordings using `build_ipn_dataset.py` (or a lightweight equivalent that reads the same JSONL format).
3. Fine-tune the best available LSTM checkpoint for 5–10 epochs with a low learning rate (lr = 1e-5 to 5e-5) and a high weight decay (1e-3) to regularize over the small dataset.
4. Evaluate on a held-out 20 % split of the new recordings.

**Expected gain.** 5–10 pp on real-webcam performance. This is the single highest-impact action after Tier 1.

**Acceptance criteria.** Swipe recall ≥ 0.85 on a 5-minute real-webcam evaluation session (logged via inference metrics).

### T2-B — Switch Dataset Builder to `last` Label Strategy

**Problem.** `label_strategy: majority` assigns a gesture label to any 24-frame window where the gesture occupies slightly more than 50 % of frames. For a 300 ms swipe inside an 800 ms window, this means the model is trained to classify windows where the majority of frames are neutral rest-state poses as a swipe. Inference windows share the same structure, but the ambiguous framing hurts precision.

**Plan.**

1. In `configs/model/lstm_default.yaml` (and the IPN dataset build script), set `label_strategy: last`.
2. Also set `min_trailing_label_purity: 0.95` so only windows whose final 95 % of frames share the labeled class are accepted into training. This discards transition windows and centers training on windows that end inside a clean gesture.
3. Rebuild the full IPN dataset NPZ and retrain from scratch.

**Expected gain.** Improved swipe precision; reduced confusion between gesture onset frames and the gesture itself. Estimated +2–4 pp on test macro F1.

**Note.** This change requires a full dataset rebuild and full retraining. Do not combine with T2-A fine-tuning — run as a separate experiment.

### T2-C — Motion Energy Pre-Filter in Inference Pipeline

**Problem.** The classifier receives and processes every sliding window, including windows where the hand is completely stationary. Stationary windows occasionally produce low-confidence false classifications that pollute the EMA state and trigger spurious votes.

**Plan.**

1. In `PredictionStabilizer.update` (or a pre-processing step in the inference loop), compute the L2 norm of the `wrist_vel_raw` feature block (dims 144–147) over all frames in the current window.
2. If the maximum per-frame norm is below a threshold (empirically tune starting at 0.03 camera units/frame), emit `unknown` directly without invoking the model.

**Implementation note.** The pre-filter must receive the raw feature array, not just the `Prediction` entity. The cleanest integration point is in `run_realtime.py` before the model call, or as an early-return guard in the `GestureRecognitionPipeline.process()` method.

**Expected gain.** Reduced false triggers; improved `unknown` class precision.

---

## Tier 3 — Feature Engineering (Dataset Rebuild + Retrain)

> **Prerequisite**: Tier 2 completed and new dataset NPZ available.

### T3-A — Per-Feature Z-Score Normalization

**Problem.** `wrist_vel_raw` (dims 144–147) lives in camera-normalized [0, 1] space and has substantially higher variance than the wrist-relative velocity block (dims 71–134), which is small-scale by construction. The LSTM gates receive disproportionately large gradients from 3 of 147 features, creating an implicit feature-importance bias that cannot be corrected by learned weights alone.

**Plan.**

1. During dataset build, compute per-feature mean and standard deviation across all training windows.
2. Save the statistics alongside the NPZ (e.g., `feature_stats.npz` with keys `mean` and `std`).
3. Apply z-score normalization in `LandmarkFeatureTransformer` at inference time (load stats from the artifact bundle or alongside the model checkpoint).
4. Retrain from scratch with normalized features.

**Expected gain.** Equalized gradient contributions; better use of the model's capacity. Estimated +1–3 pp on swipe and zoom classes.

### T3-B — Temporal Acceleration Features

**Problem.** The current feature schema captures velocity (first derivative of position) but not acceleration (second derivative). Acceleration produces a sharp signed peak at gesture onset and trough at gesture offset, which is a discriminative signal for gesture segmentation that the current features cannot represent.

**Plan.**

1. In `feature_engineering.py`, add a 3-dim acceleration block: the frame-over-frame change in `wrist_vel_raw`.
2. Add the block as dims 147–150 (keeping backward compatibility via a feature schema version bump to v6).
3. Update `input_size` in `configs/model/lstm_default.yaml` from 147 to 150.
4. Rebuild dataset and retrain.

**Expected gain.** Improved onset/offset discrimination; reduced confusion between gesture preparation and gesture execution frames.

---

## Tier 4 — Architecture Changes

> **Prerequisite**: Tiers 1–3 validated. Architecture changes amplify improvements from better data; they do not substitute for fixing the data pipeline.

### T4-A — Two-Stream Architecture (Pose + Motion)

**Motivation.** The 6 directional dims (`wrist_traj` + `wrist_vel_raw`) are 4 % of the 147-dim input. The Bahdanau attention attends over time steps but cannot selectively amplify those 6 dims within each time step. A two-stream model gives the motion features a dedicated representational pathway.

**Design.**

```
Stream A (Pose):   LSTM(hidden=64,  input=73)   ← positions[0:63] + angles[134:144]
Stream B (Motion): LSTM(hidden=64,  input=74)   ← velocity[71:134] + wrist_vel[144:147] + traj[68:71]
Fusion:            concat(h_A, h_B) → Dropout → Linear(128, 8)
```

Total parameters ≈ 200 K (similar to current 264 K). Each stream is independently regularizable.

**Integration.** The forward interface remains `(B, T, F) → logits`. The model serialization format changes (two LSTM modules); update `export_checkpoint.py` and `checkpoint_loader.py` accordingly.

**Expected gain.** +4–7 pp on swipe recall versus the best Tier 3 model.

### T4-B — ST-GCN Lightweight (Phase 3 Only)

**Motivation.** Spatial Graph Convolutional Networks (Yan et al., AAAI 2018) model the hand as a 21-node kinematic graph with anatomical adjacency. Graph convolution explicitly encodes parent-child relationships (wrist → palm → finger segments), natively capturing pinch (edge contraction) and swipe (global node displacement) without relying on flat-vector feature engineering.

**Design.** 2-layer spatial GCN per frame + 1D temporal convolution across the 24-frame sequence + global average pool + classification head. Estimated parameters: 350–400 K.

**Integration cost.** High. The feature pipeline must produce graph tensors `(B, T, 21, 3)` instead of flat vectors `(B, T, F)`. The entire `LandmarkFeatureTransformer` and dataset builder must be rewritten for graph I/O.

**Expected gain.** +5–8 pp accuracy ceiling above Tier 3, based on published benchmarks on NTU RGB+D and SHREC-17 with MediaPipe skeleton input.

**Recommended scope.** Reserve for a dedicated development phase after Tier 1–3 benefits are validated and user-calibration data is available.

---

## Experiment Tracking

Each tier should be run as a separate named experiment in `experiments/runs/` with a `dataset_metadata.json` and `training_metrics.json`. The tier letter and date should appear in the experiment ID comment in `config_snapshot.yaml`.

When comparing runs, the canonical metrics are:

- **Val macro F1** (primary) — model quality
- **Real-webcam swipe recall** (secondary) — deployment quality
- **Test macro F1** (reference) — IPN generalization

The real-webcam recall is more important than test macro F1 for this use case. A model that achieves 75 % IPN test accuracy but 90 % real-webcam swipe recall is strictly preferable to one with 82 % IPN test accuracy and 60 % real-webcam swipe recall.
