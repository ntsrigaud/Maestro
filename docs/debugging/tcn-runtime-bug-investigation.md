# TCN Runtime Bug Investigation

**Branch**: `build/model-improvement`  
**Symptom**: `run_realtime.py` predicts only `unknown` for all gestures after the TCN enhancement plan was implemented.

---

## Bugs Found

### Bug 1 — Architecture mismatch (critical)

**Root cause**: `export_checkpoint.py` saved `model_config` without `num_channels` or `kernel_size` for the TCN. The checkpoint loader defaulted to `num_channels=[64,64,64]` (3 blocks) instead of the trained `[64,64,64,64]` (4 blocks). With `strict=False` the mismatch was silently accepted — `network.3.*` keys were discarded.

**Evidence**:

```
keys in artifact after training:  network.{0,1,2,3}.*, head.*
keys used at runtime:              network.{0,1,2}.*, head.*  ← 4th block absent
unexpected_keys (strict=False):  ['network.3.net.0.weight', 'network.3.net.0.bias', ...]
missing_keys:                    []   ← head loaded correctly, so no error raised
```

**Fixes**:

- `export_checkpoint.py`: conditionally adds `num_channels`/`kernel_size` to `model_config` for TCN artifacts.
- `checkpoint_loader.py`: raises `ValueError` if `num_channels` is absent when loading a TCN artifact (no silent fallback). `strict=True` restored — architecture mismatches now hard-fail.
- Artifact `models/exported/tcn_v1_20260413T065716Z_inference.pt` patched in-place.

---

### Bug 2 — Unbounded wrist trajectory accumulation (moderate)

**Root cause**: `LandmarkFeatureTransformer` stored `_first_wrist: list | None`. During training, a new transformer is instantiated for each recording in `_reset_recording_state()`, so `_first_wrist` is always relative to the first frame of that recording. At runtime the instance persists for the entire session; the wrist-trajectory feature drifts unboundedly as the session origin diverges from the current hand position.

**Evidence**: Simulation of 30 s at 30 fps showed max trajectory magnitude ~0.086 at session start vs bounded max ~0.076 with a window deque — a modest but consistent bias to non-zero values not seen during training.

**Fix**: `_first_wrist` replaced with `_wrist_history: deque(maxlen=window_length)` (default 24 frames). `origin = _wrist_history[0]` — trajectory is always relative to the oldest frame in the current window.

---

### Bug 3 — `strict=False` silently masking weight load errors (moderate)

**Root cause**: A prior commit changed `model.load_state_dict(state_dict)` to `model.load_state_dict(state_dict, strict=False)` in `checkpoint_loader.py`. This suppressed the `RuntimeError` that PyTorch would otherwise raise for unexpected keys, converting a detectable hard failure into silent partial weight loading.

**Fix**: Reverted to `model.load_state_dict(state_dict)` (strict=True by default).

---

## Why Only `unknown` Was Predicted

The 3-block model (Bug 1) outputs heavily biased class logits:

- `unknown` logit consistently dominates (27–43% softmax probability)
- No gesture class reaches the runtime confidence threshold (0.80–0.90)
- Result: every frame classifies as `unknown` or is suppressed

The LSTM baseline (78.8% test accuracy) was unaffected; only the TCN path triggered this failure.

---

## Fixes Applied

| File                                                   | Change                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `src/.../training/export_checkpoint.py`                | Add `num_channels`/`kernel_size` to TCN `model_config` on export                    |
| `src/.../infrastructure/model/checkpoint_loader.py`    | Raise `ValueError` on missing `num_channels`; restore `strict=True`                 |
| `src/.../pipeline/feature_engineering.py`              | Replace `_first_wrist` with `deque(maxlen=window_length)`                           |
| `models/exported/tcn_v1_20260413T065716Z_inference.pt` | Patched artifact payload with correct `num_channels=[64,64,64,64]`, `kernel_size=3` |

---

## Post-fix Validation

```
ruff:   All checks passed (4 errors fixed, 0 remaining)
mypy:   Success: no issues found in 63 source files
pytest: 189 passed, 14 deselected in 2.34s
```

Loader verification (Python):

```python
artifact = load_inference_artifact("models/exported/tcn_v1_20260413T065716Z_inference.pt", device=cpu)
# GestureTCNClassifier, feature_dim=144
len(list(artifact.model.network.children()))  # → 4
```

---

## Outstanding Notes

The TCN test accuracy (75.0%) is below the LSTM baseline (78.8%). The runtime correctness fix (Bug 1) is necessary but may not close the quality gap. Possible follow-up investigations:

1. Evaluate whether the session-start wrist trajectory bias (Bug 2) during training affected learned representations — retrain with the deque fix to confirm.
2. Compare LSTM+attention (Phase 1 only, 134-dim) vs TCN (144-dim) on the same dataset split.
3. Consider promoting the LSTM model as the production deployment while investigating the TCN quality gap.
