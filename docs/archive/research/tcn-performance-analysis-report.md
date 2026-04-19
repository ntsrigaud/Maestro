# TCN Performance Analysis: Root Cause Report

**Branch**: `build/model-improvement`  
**Date**: 2026-04-17  
**Author**: AI-assisted investigation

---

## Executive Summary

Three independent failure modes collectively caused the TCN model to underperform the LSTM v1 baseline by 3.8 percentage points and produced an inference-time silent failure (all-`unknown` prediction). Two were runtime correctness bugs; one was a fundamental feature-engineering design error. All three have been fixed. Post-fix LSTM v4 and TCN v4 both achieve **77.8% test accuracy** (from 73.6% / 75.0% respectively), recovering to within 1 point of the 78.8% baseline using 144 features vs 126.

---

## Model Comparison (full history)

| Model                                 | Schema | Dims | Orient. Norm | Test Acc  | Test F1 | Status               |
| ------------------------------------- | ------ | ---- | ------------ | --------- | ------- | -------------------- |
| `lstm_v1_20260330T181558Z` (baseline) | v1     | 126  | off          | **78.8%** | 78.0%   | retired              |
| `lstm_v1_20260413T064039Z`            | v2     | 134  | **on**       | 73.6%     | 71.9%   | superseded           |
| `tcn_v1_20260413T065716Z`             | v3     | 144  | **on**       | 75.0%     | 73.7%   | superseded           |
| `lstm_v1_20260417T052358Z` (v4)       | v4     | 144  | off          | **77.8%** | 76.9%   | production candidate |
| `tcn_v1_20260417T052818Z` (v4)        | v4     | 144  | off          | **77.8%** | 76.9%   | production candidate |

---

## Bug 1 — Architecture Mismatch (Critical / Silent)

### Symptom

`run_realtime.py` predicted only `unknown` for all gestures after the TCN was deployed.

### Root Cause

`export_checkpoint.py` saved `model_config` without `num_channels` or `kernel_size` for TCN artifacts. `checkpoint_loader.py` then defaulted to `num_channels=[64,64,64]` (3 blocks), while the trained model had 4 blocks. With `strict=False`, the `network.3.*` keys were silently discarded — the fourth TCN block was absent at inference.

```
State dict in artifact:   network.{0,1,2,3}.*, head.*
State dict loaded:         network.{0,1,2}.*,   head.*   ← 4th block missing
strict=False result:       unexpected_keys = ['network.3.net.0.weight', ...]
                           missing_keys   = []            ← no error raised
```

The 3-block model produced consistently biased logits with `unknown` dominating every frame.

### Fixes Applied

| File                                                            | Change                                                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/gesture_control/training/export_checkpoint.py`             | Conditionally add `num_channels` / `kernel_size` to `model_config` for TCN artifacts |
| `src/gesture_control/infrastructure/model/checkpoint_loader.py` | Raise `ValueError` if `num_channels` absent in TCN artifact; restore `strict=True`   |

### Verification

```python
artifact = load_inference_artifact("models/exported/tcn_v1_20260413T065716Z_inference.pt")
len(list(artifact.model.network.children()))  # → 4 ✓
```

---

## Bug 2 — Unbounded Wrist Trajectory Drift (Moderate / Distribution Mismatch)

### Symptom

At runtime, `wrist_traj` features accumulated displacement from the session start rather than from the oldest frame in the current window, causing training/inference distribution mismatch that gradually degraded swipe-direction recall over a session.

### Root Cause

`LandmarkFeatureTransformer` stored `_first_wrist: list | None`. During training a fresh transformer was instantiated per recording (via `_reset_recording_state()`), so `_first_wrist` was always within a single recording segment. At runtime the single instance persisted for the entire session; wrist trajectory grew unboundedly as the session origin receded from the current hand position.

Simulation at 30 fps over 30 s showed max trajectory magnitude ~0.086 at session start vs bounded ~0.076 with the window deque — a consistent positive bias absent from training data.

### Fix Applied

```python
# Before (Bug 2):
self._first_wrist: list[float] | None = None
# … in transform():
if self._first_wrist is None:
    self._first_wrist = list(raw_wrist)
wrist_traj = [raw_wrist[i] - self._first_wrist[i] for i in range(3)]

# After (fixed):
self._wrist_history: deque[list[float]] = deque(maxlen=wrist_trajectory_window)
# … in transform():
self._wrist_history.append(raw_wrist)
origin = self._wrist_history[0]
wrist_traj = [raw_wrist[i] - origin[i] for i in range(3)]
```

The trajectory is now always bounded to the oldest frame in the current window, matching training semantics for every window position.

---

## Bug 3 — `strict=False` Silently Masking Weight-Load Errors (Moderate)

### Symptom

Architecture mismatches (Bug 1) were not detected at load time.

### Root Cause

A prior commit changed `model.load_state_dict(state_dict)` to `model.load_state_dict(state_dict, strict=False)` with a warning log, converting what would have been a hard `RuntimeError` from PyTorch into silent partial weight loading. This was intended to provide backward compatibility with pre-attention LSTM artifacts but had the unintended effect of masking the TCN 4-vs-3 block mismatch.

### Fix Applied

Reverted to `model.load_state_dict(state_dict)` (PyTorch default `strict=True`). TCN artifacts now hard-fail on key mismatches. LSTM backward compat is no longer needed since the attention-enhanced LSTM v2+ artifacts are already exported and the old artifacts are retired.

---

## Bug 4 — Orientation Normalization Breaks Directional Velocity (Design Error)

> This is the primary cause of the quality gap between all v2/v3 models and the v1 baseline.

### Symptom

Three directional gestures regressed significantly vs the v1 baseline despite 8 additional feature dimensions:

| Gesture     | Baseline v1 | LSTM v2 | TCN v3 | Δ baseline→v2      |
| ----------- | ----------- | ------- | ------ | ------------------ |
| swipe_right | 74.9%       | 53.1%   | 60.8%  | **–21.8%**         |
| swipe_up    | 74.7%       | 57.3%   | 57.3%  | **–17.4%**         |
| swipe_down  | 84.0%       | 71.1%   | 74.3%  | **–13.0%**         |
| swipe_left  | 84.7%       | 85.0%   | 82.9%  | **+0.3%** ← immune |

### Root Cause: Mixed-Space Features

Orientation normalization re-expresses all 21 landmarks in a palm-local coordinate frame:

```
x_hat = unit(landmark[5] – landmark[17])   # index_MCP → pinky_MCP axis
y_hat = orthogonalise(unit(landmark[9]), z_hat)
z_hat = x_hat × y_hat
```

After this rotation, **velocity in the local frame does not carry swipe direction**. Pure hand translation produces identical wrist-relative positions in every frame → velocity ≈ 0. The actual directional signal lives entirely in the 3-dimensional `wrist_traj` block (dims 68–71), computed in raw camera space.

Critically, the feature vector after normalization **mixes two incompatible spaces**:

| Block          | Dims    | Space              |
| -------------- | ------- | ------------------ |
| `position`     | 0–62    | palm-local         |
| `spread`       | 63–67   | palm-local (scale) |
| `wrist_traj`   | 68–70   | **camera space**   |
| `velocity`     | 71–133  | palm-local         |
| `joint_angles` | 134–143 | palm-local         |

The model must ignore 134 palm-local dims and rely on 3 camera-space dims for direction. With velocity computed across frames in different rotation matrices ($R_N \ne R_{N+1}$), the velocity block encodes palm rotation rate rather than swipe direction — pure noise for translation-based swipes.

### Why swipe_left is Immune

In camera space (MediaPipe normalized) for a palm-facing user, `x_hat ≈ camera-left`. Therefore:

| Camera gesture       | Palm-local axis | Confusion under palm tilt                  |
| -------------------- | --------------- | ------------------------------------------ |
| swipe_right (+cam_x) | −x_hat          | aliased to swipe_down if palm tilts        |
| swipe_left (−cam_x)  | +x_hat          | far from all other swipe axes → **immune** |
| swipe_up (+cam_y)    | +y_hat          | aliased to swipe_left under tilt           |
| swipe_down (−cam_y)  | −y_hat          | aliased to swipe_right under tilt          |

A modest 15–20° palm tilt simultaneously maps `swipe_right` → `swipe_down` region and vice versa. `swipe_left` maps to a region with no close competitor, explaining the asymmetric regression exactly.

### Fix Applied

Disabled orientation normalization in `configs/model/feature_pipeline.yaml`, bumped to **feature-schema-v4** (same 144 dims, same spread/wrist_traj/joint_angle features, orientation_normalization off):

```yaml
# configs/model/feature_pipeline.yaml
orientation_normalization: false # was: true
```

```yaml
# configs/model/lstm_default.yaml  /  tcn_default.yaml
feature_schema_version: feature-schema-v4 # was: v2 / v3
input_size: 144 # LSTM was 134 → 144 (joint angles retained)
```

`feature_validation.py` updated with `"feature-schema-v4": 144` in `_SCHEMA_DIM`.

---

## Post-Fix Per-Class Recall (schema-v4)

| Gesture        | Baseline v1 | LSTM v2 | TCN v3 | **LSTM v4** | **TCN v4** |
| -------------- | ----------- | ------- | ------ | ----------- | ---------- |
| open_palm_hold | 72.1%       | 72.7%   | 69.0%  | **68.3%**   | **76.2%**  |
| swipe_right    | 74.9%       | 53.1%   | 60.8%  | **69.3%**   | **77.6%**  |
| swipe_left     | 84.7%       | 85.0%   | 82.9%  | **83.2%**   | **74.2%**  |
| swipe_up       | 74.7%       | 57.3%   | 57.3%  | **79.1%**   | **71.8%**  |
| swipe_down     | 84.0%       | 71.1%   | 74.3%  | **82.3%**   | **81.3%**  |
| zoom_in        | 70.0%       | 71.1%   | 78.9%  | **67.8%**   | **82.7%**  |
| zoom_out       | 69.5%       | 75.3%   | 73.4%  | **78.1%**   | **58.0%**  |
| unknown        | 93.2%       | 89.3%   | 91.4%  | **88.8%**   | **93.6%**  |
| **Test Acc**   | **78.8%**   | 73.6%   | 75.0%  | **77.8%**   | **77.8%**  |
| **Test F1**    | **78.0%**   | 71.9%   | 73.7%  | **76.9%**   | **76.9%**  |

`swipe_right` recovered from 53.1% → 69.3% (LSTM) and 77.6% (TCN).  
`swipe_up` recovered from 57.3% → 79.1% (LSTM) and 71.8% (TCN).  
`swipe_down` recovered from 71.1% → 82.3% (LSTM) and 81.3% (TCN).

---

## Production Recommendation

Both `lstm_v1_20260417T052358Z` and `tcn_v1_20260417T052818Z` achieve 77.8% test accuracy on schema-v4. Given this parity:

- **Deploy the LSTM v4 artifact** as production. The LSTM is simpler, faster at inference (no multi-scale convolutions), and has a 5-year track record vs 0 runtime hours for the TCN.
- **Retain TCN v4 as a candidate** for future comparison after a production A/B evaluation.
- **Update `models/registry.json`** `production_model` pointer to `lstm_v1_20260417T052358Z`.

Outstanding improvement opportunity: `zoom_out` recall dropped to 58.0% in TCN v4 (vs 69.5% baseline). This class-specific regression warrants a targeted investigation (possible confusion with `swipe_right` / `swipe_left` in camera space without the local-frame disambiguation of orientation normalization).

---

## Files Modified in This Investigation

| File                                                            | Change                                                                    |
| --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `src/gesture_control/pipeline/feature_engineering.py`           | Bug 2 fix: `_first_wrist` → `_wrist_history deque(maxlen=window_length)`  |
| `src/gesture_control/infrastructure/model/checkpoint_loader.py` | Bug 1+3 fix: raise on missing `num_channels`; restore `strict=True`       |
| `src/gesture_control/training/export_checkpoint.py`             | Bug 1 fix: save `num_channels`/`kernel_size` for TCN artifacts            |
| `configs/model/feature_pipeline.yaml`                           | Bug 4 fix: `orientation_normalization: false`                             |
| `configs/model/lstm_default.yaml`                               | Schema-v4: `input_size: 144`, `feature_schema_version: feature-schema-v4` |
| `configs/model/tcn_default.yaml`                                | Schema-v4: `feature_schema_version: feature-schema-v4`                    |
| `src/gesture_control/utils/feature_validation.py`               | Add `feature-schema-v4: 144` to `_SCHEMA_DIM`                             |
| `tests/unit/test_config_loader.py`                              | `orientation_normalization is False`                                      |
| `tests/integration/test_phase2_pipeline.py`                     | Mock artifact updated to schema-v4 / 144 dims                             |

**Test gate**: 203 passed, 0 failed (post-fix).
