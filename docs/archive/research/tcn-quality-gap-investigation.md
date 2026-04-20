# TCN Quality Gap Investigation

**Branch**: `build/model-improvement`  
**Question**: Why does the new TCN (75.0% test accuracy) and LSTM v2 (73.6%) under-perform the LSTM v1 baseline (78.8%)?

---

## Model Comparison

| Model                                 | Schema | Dims | Orient. Norm | Test Acc  | Test F1 |
| ------------------------------------- | ------ | ---- | ------------ | --------- | ------- |
| `lstm_v1_20260330T181558Z` (baseline) | v1     | 126  | off          | **78.8%** | 78.0%   |
| `lstm_v1_20260413T064039Z`            | v2     | 134  | **on**       | 73.6%     | 71.9%   |
| `tcn_v1_20260413T065716Z`             | v3     | 144  | **on**       | 75.0%     | 73.7%   |

All three models were trained on the same 22,973 IPN windows with identical label distributions. The only differences are the feature schema version and orientation normalization setting.

---

## Per-Class Recall Analysis

| Gesture         | Baseline (v1) | LSTM v2 (v2) | TCN v1 (v3) | Δ (baseline→v2) |
| --------------- | ------------- | ------------ | ----------- | --------------- |
| open_palm_hold  | 72.1%         | 72.7%        | 69.0%       | +0.6%           |
| **swipe_right** | **74.9%**     | **53.1%**    | **60.8%**   | **–21.8%**      |
| swipe_left      | 84.7%         | 85.0%        | 82.9%       | +0.3%           |
| **swipe_up**    | **74.7%**     | **57.3%**    | **57.3%**   | **–17.4%**      |
| **swipe_down**  | **84.0%**     | **71.1%**    | **74.3%**   | **–13.0%**      |
| zoom_in         | 70.0%         | 71.1%        | 78.9%       | +1.1%           |
| zoom_out        | 69.5%         | 75.3%        | 73.4%       | +5.8%           |
| unknown         | 93.2%         | 89.3%        | 91.4%       | –4.0%           |

**Finding**: Only three-quarters of the directional gestures regress significantly. `swipe_left` is a striking exception — broadly unchanged. Non-directional gestures (zoom, static holds) are unaffected or slightly improved.

---

## Root Cause: Orientation Normalization Breaks Velocity Coherence

### How orientation normalization works

Each frame's landmarks are re-expressed in a hand-local coordinate frame:

```
x_hat = unit(landmark[5] – landmark[17])   # index_MCP → pinky_MCP axis
y_hat = orth(unit(landmark[9]), z_hat)      # toward middle-MCP, orthogonalised
z_hat = x_hat × y_hat
```

All 21 landmarks are projected onto `(x_hat, y_hat, z_hat)` before any velocity or trajectory computation.

### Why this degrades directional-gesture recognition

**Step 1 — velocity in wrist-relative space is zero for pure translation.**  
After the wrist is subtracted from all landmarks, a pure hand translation (swipe) produces identical wrist-relative positions in every frame → velocity features ≈ 0. The velocity block (dims 71–134 in v2) therefore does not carry swipe direction regardless of normalization.

**Step 2 — the directional signal lives entirely in `wrist_traj` (dims 68–71).**  
`wrist_traj` is computed in raw camera space: `[wrist_now – wrist_window_origin]`. This 3-dim feature IS the primary carrier of swipe direction for all models in feature-schema-v2/v3.

**Step 3 — orientation normalization re-expresses velocity in a frame that rotates between frames.**  
Even though velocity is near-zero for pure translation, real swipes include slight palm rotation. With normalization, the velocity at frame N is expressed in `R_N`, and at frame N+1 in `R_{N+1} ≠ R_N`. The velocity computed as `R_{N+1}·H – R_N·H` encodes palm rotation rate, not swipe direction. Different users hold their hand at different tilts, so the same camera-space gesture produces different velocity patterns in the palm-local frame — increasing intra-class variance.

**Step 4 — coordinate-frame mixing forces the model to learn cross-space features.**  
After normalization, the feature vector mixes two incompatible spaces:

- `position[0:63]` / `velocity[71:134]` — palm-local frame
- `wrist_traj[68:71]` — camera frame

The model must learn to ignore the mixed-space velocity noise and rely almost entirely on 3 wrist_traj dims for direction. With only 3 discriminating dims vs 126 total in v1, the learned boundary is noisier.

### Why swipe_right is hit hardest; swipe_left is immune

In camera space (MediaPipe normalized coordinates), for a user with palm facing the camera:

```
x_hat ≈ camera-left   (index_MCP is camera-right of pinky_MCP)
y_hat ≈ camera-up
```

| Camera gesture       | In palm-local frame | In-frame alias                                           |
| -------------------- | ------------------- | -------------------------------------------------------- |
| swipe_right (+cam_x) | −x_hat direction    | near −y_hat if palm tilts → **confused with swipe_down** |
| swipe_left (−cam_x)  | +x_hat direction    | clean, far from other swipe axes                         |
| swipe_up (+cam_y)    | +y_hat direction    | near +x_hat if palm tilts → confused with swipe_left     |
| swipe_down (−cam_y)  | −y_hat direction    | near −x_hat → confused with swipe_right                  |

A modest palm tilt simultaneously maps swipe_right → swipe_down and vice-versa; swipe_left maps into a region with no close neighbour. This explains the asymmetry exactly:

```
swipe_right confusion (baseline → lstm_v2):
  confused with open_palm_hold:  4.7% → 12.1%  (+7.4%)
  confused with swipe_down:      3.8% →  9.7%  (+5.9%)
  confused with unknown:         5.3% → 10.9%  (+5.6%)
```

---

## Secondary Confound: `wrist_traj` Training–Inference Semantic Mismatch

During training, `LandmarkFeatureTransformer` is instantiated fresh for every recording segment via `_reset_recording_state()`. The wrist trajectory origin (`_first_wrist`) is therefore the **recording start**. For mid-recording windows, wrist_traj accumulates the total displacement from recording start, which can be large.

After the Bug #2 fix (`deque(maxlen=24)`), runtime inference computes wrist_traj relative to the **oldest frame in the current window** – always bounded to ≤ 24 frames of motion.

For the first window of each recording the two semantics are identical. For later windows, training data has larger trajectory values than the model will ever see at inference. This distribution mismatch degrades swipe-direction precision further, and is present in all models trained on ipn_dataset_v2/v3.

---

## Recommended Fix: `feature-schema-v4`

Rebuild the dataset with **orientation_normalization disabled** while retaining all new features. This recovers camera-space position information as an implicit directional cue while keeping spread, wrist_traj, and joint_angle benefits.

| Schema            | Dims    | Orient. Norm | Spread | Wrist Traj | Joint Angles |
| ----------------- | ------- | ------------ | ------ | ---------- | ------------ |
| v1 (baseline)     | 126     | off          | —      | —          | —            |
| v2                | 134     | **on**       | ✓      | ✓          | —            |
| v3                | 144     | **on**       | ✓      | ✓          | ✓            |
| **v4 (proposed)** | **144** | **off**      | ✓      | ✓          | ✓            |

Expected outcome: retain spread/wrist_traj/joint_angle gains while recovering the directional recall lost to orientation normalization. Projected accuracy ≥ 78% (matching or exceeding baseline).

### Execution sequence

1. Update `configs/model/feature_pipeline.yaml`: set `orientation_normalization: false`, bump `feature_schema_version` to `feature-schema-v4`.
2. Rebuild IPN dataset: `python scripts/build_ipn_dataset.py`.
3. Retrain LSTM: `python scripts/train_model.py --dataset <v4 path>`.
4. Evaluate. If test accuracy ≥ 78%, retrain TCN on v4 dataset.
5. Update `production_model` in `models/registry.json`.

> **Note**: The wrist_traj semantic mismatch (Bug #2) is already fixed in `feature_engineering.py`. Rebuilding the dataset will automatically use the corrected deque semantics.

---

## Production Status

While investigating, `lstm_v1_20260413T064039Z` (73.6% test acc, schema v2) is pinned as production via `models/registry.json → production_model`. It uses the same feature pipeline as the broken TCN, so its runtime trajectory is consistent.
