# Work Log — Real-Time Performance Enhancement: EMA Bypass, Two-Stream LSTM, and Improvement Roadmap

**Date:** 2026-04-20  
**Branch:** `refactor/enhance-real-time-performance`  
**Author:** @ntsrigaud

---

## Context

Despite 79 % test accuracy on the IPN Hand held-out set, the system failed to reliably
dispatch swipe gestures in real-world webcam use. Root-cause analysis confirmed five
distinct failure modes, of which two are critical: the `PredictionStabilizer` EMA
ramp-up penalty (500 ms to cross the dispatch threshold for a 300 ms gesture) and a
domain gap between IPN lab recordings and real-webcam conditions. This branch delivers
the complete Tier 1 post-processing remediation from `model-improvement-plan.md`,
integrates the Tier 4-A two-stream LSTM architecture (via PR #157), and establishes the
structured improvement roadmap document.

---

## Scope of Work

### Inference Post-Processing — Tier 1

- Introduced `_ema_bypass_labels` in `PredictionStabilizer` to derive the set of
  instant-dispatch classes (those with `vote_window == 1`) at construction time.
- Modified `_smooth_probs()` to return raw model probabilities for bypass-label classes
  while continuing to advance the internal EMA state — preserving full smoothing for
  held and zoom gestures.
- Added `tests/unit/test_prediction_stabilization.py` (74 lines) with two test cases
  covering bypass-label membership and EMA attenuation-bypass behaviour.
- Applied a diagnostic threshold reduction in `configs/thresholds/production_ipn.yaml`
  (0.80–0.90 → 0.70–0.75) with an inline `DIAGNOSTIC` comment and documented revert
  condition.

### Architecture — Two-Stream LSTM (PR #157)

- Implemented `TwoStreamGestureLSTM` in
  `src/gesture_control/infrastructure/model/two_stream_lstm.py` (150 lines): the 147-dim
  feature-schema-v5 input is partitioned into a Pose stream (73 dims: positions,
  joint angles) and a Motion stream (74 dims: fingertip spread, wrist trajectory,
  velocity, wrist_vel_raw), each processed by an independent 2-layer LSTM + Bahdanau
  temporal attention; context vectors are fused via concatenation and a linear
  classification head.
- Extended `checkpoint_loader.py`, `train_lstm.py`, and `export_checkpoint.py` with
  dispatch branches for `two_stream` model names.
- Added `configs/model/two_stream_lstm.yaml` with the full training configuration.
- Promoted `two_stream_lstm_v1_20260420T091839Z_inference.pt` as the production artifact
  in `configs/model_source.yaml`.

### Documentation and Planning

- Authored `docs/development/model-improvement-plan.md` (176 lines) — tiered improvement
  roadmap organised into four tiers (post-processing, data pipeline, feature engineering,
  architecture); each tier includes problem statement, implementation plan, expected gain,
  and acceptance criteria.
- Updated `docs/architecture/data-flow.md`, `docs/architecture/logical-view.md`, and
  `docs/ml-system/feature-schema-v5.md` to reflect the two-stream checkpoint loading path
  and EMA bypass behaviour.

---

## Technical Changes

**`src/gesture_control/pipeline/stabilization.py`**
`_ema_bypass_labels: frozenset[str]` derived at construction time from `class_vote_windows`.
`_smooth_probs()` now maintains two outputs per frame: `next_ema` (always updated) and
`output_probs` (raw probability for bypass classes, EMA for all others). Eliminates the
~5-frame ramp-up delay that previously attenuated a 0.95 raw swipe confidence to 0.38 on
the first frame.

**`src/gesture_control/infrastructure/model/two_stream_lstm.py`** (new)
`TwoStreamGestureLSTM.forward(x: Tensor[B, T, 147]) → logits[B, 8]` — unchanged
interface from `GestureLSTMClassifier`. `_split()` partitions by semantic feature block.
`_attend()` applies a learned `Linear(H, 1)` score over the time axis and returns the
softmax-weighted sum of hidden states. Parameter count ≈ 143 K vs ≈ 284 K for the
single-stream hidden=128 baseline.

**`src/gesture_control/infrastructure/model/checkpoint_loader.py`**
New dispatch branch: `artifact_type.startswith("two_stream") or model_name.startswith("two_stream")` → `TwoStreamGestureLSTM`.

**`src/gesture_control/training/train_lstm.py`**
`_build_model()` dispatches on `model_name.startswith("two_stream")` → `TwoStreamGestureLSTM`.

**`src/gesture_control/training/export_checkpoint.py`**
`artifact_type` resolves to `"two_stream_lstm_inference"` for `two_stream` prefix; falls
through to `"lstm_inference"` otherwise.

**`configs/thresholds/production_ipn.yaml`**
All class thresholds reduced from 0.80–0.90 to 0.70–0.75 (diagnostic T1-B). Inline
comment specifies the revert condition and target range (0.75–0.80).

---

## Training Results

Three two-stream training runs were executed on feature-schema-v5
(`ipn_dataset_v5_mirrored.npz`); the second run was promoted:

| Run                                          | Test Acc | Macro-F1 | Best Epoch |
| -------------------------------------------- | -------- | -------- | ---------- |
| `two_stream_lstm_v1_20260420T073729Z`        | 0.7917   | 0.7836   | 12         |
| `two_stream_lstm_v1_20260420T091431Z`        | 0.7846   | 0.7750   | 8          |
| `two_stream_lstm_v1_20260420T091839Z` (prod) | 0.7899   | 0.7795   | 11         |

Compared to the single-stream v5 baseline (`lstm_v1_20260418T023120Z`, acc=0.7907,
macro-F1=0.7807), the best two-stream run (073729Z) improves `swipe_down` (+5.7 pp),
`zoom_out` (+4.9 pp), `zoom_in` (+2.5 pp), and `unknown` (+2.7 pp), at approximately
50 % of the parameter budget. See `docs/archive/work-logs/2026-04-20-two-stream-lstm.md`
for the full per-class breakdown and improvement trajectory table.

---

## Testing and Validation

- `tests/unit/test_prediction_stabilization.py` — two new test cases added; EMA bypass
  membership and attenuation-bypass behaviour verified.
- `tests/unit/test_adapters.py` — Black formatter applied; no logic changes.
- Notebook linting violations resolved with Ruff; Black formatting applied to all
  notebooks in `experiments/notebooks/`.
- No existing tests modified or suppressed.

---

## Outcome

The branch eliminates the primary real-time dispatch failure: swipe gestures are no longer
filtered by EMA ramp-up before they complete. The two-stream LSTM becomes the active
production checkpoint, offering equivalent IPN test performance at half the parameter
count while providing improved separation of motion and pose feature gradients. The
model improvement plan establishes the roadmap for remaining root causes (domain gap,
window contamination, feature normalisation, acceleration features). No breaking changes
to public interfaces; the `(B, T, F) → logits` contract is preserved across both model
types. Net change: 2,359 insertions, 31 deletions across 26 files.
