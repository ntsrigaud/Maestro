# Gesture Recognition Enhancement — Implementation Plan

## Baseline (April 2026)

| Metric        | Value                                 |
| ------------- | ------------------------------------- |
| Test accuracy | 78.8%                                 |
| Test macro F1 | 78.0%                                 |
| Feature dims  | 126                                   |
| Model readout | final hidden state `output[:, -1, :]` |
| Augmentation  | temporal scale + spatial jitter       |
| Quality gates | ✅ mypy / ruff / 204 tests            |

---

## Pre-flight Fixes

> These three infrastructure changes must land **before any Phase 1 code changes start**.
> They address blocking correctness gaps identified in the technical review. None of them
> changes observable runtime behaviour — they are purely additive or refactoring changes.

### Fix R1 — `save_npz()` Must Propagate `feature_schema_version` _(blocking)_

**File**: `src/gesture_control/training/dataset_builder.py`

`save_npz()` currently hardcodes `"feature_schema_version": "feature-schema-v1"` in the
metadata dict regardless of which features were used to build the dataset. After Phase 1 and
Phase 2 feature changes, the stored version will always be `v1`, silently breaking schema
version checks during future compatibility validation.

Add a keyword-only parameter with a backward-compatible default to the method signature:

```python
def save_npz(
    self, output_path: str | Path, *, feature_schema_version: str = "feature-schema-v1"
) -> Path:
```

Replace the hardcoded key inside the `metadata` dict:

```python
metadata = {
    "feature_schema_version": feature_schema_version,   # was: "feature-schema-v1"
    ...
}
```

Update `build_ipn_dataset.py` to accept a `--feature-schema-version` CLI argument
(default `"feature-schema-v1"`) and forward it to `save_npz()`. During Phase 1 dataset
regeneration, pass `"feature-schema-v2"`.

**Validation**: Rebuild the existing IPN dataset and assert `meta["feature_schema_version"]`
equals the argument value passed on the command line.

---

### Fix R4 — Extract `_build_criterion()` in `train_lstm.py`

**File**: `src/gesture_control/training/train_lstm.py`

`train_lstm()` constructs `nn.CrossEntropyLoss` in two separate branches (weighted and
unweighted). Step 1.8 adds `label_smoothing=0.1` to both; having two call sites is an
error-prone maintenance hazard. Eliminate the duplication by adding a private helper
before `train_lstm()`:

```python
def _build_criterion(
    class_weights: torch.Tensor | None,
    *,
    label_smoothing: float = 0.0,
) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
```

Replace both existing `nn.CrossEntropyLoss(...)` constructions in `train_lstm()`:

```python
# weighted branch:
criterion: nn.Module = _build_criterion(class_weights)
# unweighted branch:
criterion = _build_criterion(None)
```

Step 1.8 then becomes a single-parameter change at each call site.

---

### Fix R5 — Store `model_name` in Exported Artifact

**File**: `src/gesture_control/training/export_checkpoint.py`

`export_inference_artifact()` stores `model_config` but omits `model_name`. The
`checkpoint_loader.py` factory in Step 2.0 needs to identify the architecture from stored
data without relying on the generic `artifact_type` string alone.

Add `model_name` to the `model_config` sub-dict:

```python
"model_config": {
    "model_name": config.model.model_name,   # add this field
    "input_size": config.model.input_size,
    "hidden_size": config.model.hidden_size,
    "num_layers": config.model.num_layers,
    "dropout": config.model.dropout,
    "num_classes": config.model.num_classes,
},
```

This is a one-line, backward-compatible addition. Existing loaders that do not read
`model_name` are unaffected.

---

## Phase 1 — Feature & Training Hardening

**Target**: val accuracy ≥ 85%, macro F1 ≥ 84%

### Step 1.1 — Extend `LandmarkFeatureTransformer` (feature dims 126 → 134)

**File**: `src/gesture_control/pipeline/feature_engineering.py`

**1.1.a** Add fingertip index constant at module level, below the existing imports:

```python
_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
```

**1.1.b** Add `_first_normalized` state field to `__init__`:

```python
self._first_normalized: list[list[float]] | None = None
```

**1.1.c** Reset it in `_clear_velocity_state()`:

```python
def _clear_velocity_state(self) -> None:
    self._prev_normalized = None
    self._prev_timestamp_ms = None
    self._first_normalized = None  # add this line
```

**1.1.d** In `transform()`, after the `velocity = ...` line and before building `vector`,
capture the first normalized pose and compute the two new feature blocks:

```python
# Wrist trajectory: displacement of landmark 0 from the first valid frame.
if self._first_normalized is None:
    self._first_normalized = normalized
wrist_origin = self._first_normalized[0]
wrist_traj = [
    normalized[0][0] - wrist_origin[0],
    normalized[0][1] - wrist_origin[1],
    normalized[0][2] - wrist_origin[2],
]

# Fingertip spread: 5 Euclidean distances between selected tip pairs.
tips = [normalized[i] for i in _FINGERTIP_INDICES]
spread = [
    self._distance(tips[0], tips[1]),  # thumb ↔ index
    self._distance(tips[0], tips[2]),  # thumb ↔ middle
    self._distance(tips[0], tips[3]),  # thumb ↔ ring
    self._distance(tips[0], tips[4]),  # thumb ↔ pinky
    float(sum(self._distance(tips[j], tips[k])
               for j in range(len(tips)) for k in range(j + 1, len(tips)))
          / 10.0),                     # mean pairwise spread
]

vector = self._flatten(normalized) + self._flatten(velocity) + spread + wrist_traj
```

> Feature layout after this change:
>
> - dims 0–62: position (21 × 3)
> - dims 63–125: velocity (21 × 3)
> - dims 126–130: fingertip spread (5 scalars)
> - dims 131–133: wrist trajectory (3 scalars)
> - **Total: 134 dims**

**1.1.e** Update `_zero_vector()` to return 134 zeros:

```python
@staticmethod
def _zero_vector() -> list[float]:
    return [0.0] * 134
```

**Validation**: `python -c "from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer; print('OK')"` must succeed without error.

---

### Step 1.2 — Update Feature Pipeline Config

**File**: `configs/model/feature_pipeline.yaml`

Apply both changes atomically:

```yaml
orientation_normalization: true # was: false


# Feature vector structure after Phase 1 changes:
# [position(63) | velocity(63) | spread(5) | wrist_traj(3)] = 134 dims
```

Also update the schema version comment and add the new dim total to the bottom comment block. The `feature_schema_version` is embedded per-dataset at build time; updating the comment here serves as the ground-truth reference.

---

### Step 1.3 — Update Training Config

**File**: `configs/model/lstm_default.yaml`

```yaml
model:
  input_size: 134 # was: 126
  feature_schema_version: feature-schema-v2 # was: feature-schema-v1

export:
  feature_schema_version: feature-schema-v2 # was: feature-schema-v1
```

---

### Step 1.4 — Add Label Purity Filter to `SequenceDatasetBuilder`

**File**: `src/gesture_control/training/dataset_builder.py`

> **Implementation note**: The production dataset uses `label_strategy: "last"` (the
> default), meaning `selected_label` is always the trailing frame's label — not a majority
> vote. The purity check therefore measures what fraction of the 24 window frames share the
> *trailing* label. A boundary window where the first half belongs to a different gesture
> will score below 0.9 and be correctly discarded. The parameter name reflects this
> semantics explicitly.

**1.4.a** Add `min_trailing_label_purity` parameter to `__init__` (after `min_valid_ratio`):

```python
min_trailing_label_purity: float = 0.9,
```

Store it: `self._min_trailing_label_purity = min_trailing_label_purity`

**1.4.b** In `add_frame()`, replace the unconditional `self._samples.append(sample)` block
with a purity check. After `selected_label = self._select_label(labels)`, compute and check
purity before constructing `SequenceSample`:

```python
trailing_count = labels.count(selected_label)
purity = trailing_count / len(labels)
if purity < self._min_trailing_label_purity:
    self._sample_index += 1
    return emitted          # discard this boundary window
```

Place this block immediately before the `sample = SequenceSample(...)` construction.
`self._sample_index` is intentionally incremented on discarded windows to keep sample IDs
sparse but monotonic — this is harmless.

**1.4.c** Expose the parameter in `build_ipn_dataset.py` where `SequenceDatasetBuilder` is
instantiated. Set `min_trailing_label_purity=0.9` at all call sites.

**Validation**: Run `python -m pytest tests/unit/ -q` — all unit tests must pass.

---

### Step 1.5 — Regenerate Dataset

> **Required because**: feature dims changed (126 → 134) and orientation normalization toggled.

```bash
conda activate gesture-control
python scripts/build_dataset.py        # or build_ipn_dataset.py for IPN data
```

Verify the output `.npz` metadata:

```bash
python - <<'EOF'
import numpy as np, json
d = np.load("data/processed/gesture_dataset_v2.npz", allow_pickle=True)
meta = json.loads(str(d["metadata_json"]))
assert meta["feature_dim"] == 134, f"Expected 134, got {meta['feature_dim']}"
assert meta["enable_orientation_normalization"] is True
print("Dataset OK:", meta)
EOF
```

**Gate**: This verification must pass before proceeding to training.

---

### Step 1.6 — Add Temporal Attention to `GestureLSTMClassifier`

**File**: `src/gesture_control/infrastructure/model/lstm_model.py`

Add the attention head in `__init__` (after `self.head`):

```python
self.attn = nn.Linear(hidden_size, 1)
```

Replace the `forward` body:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    output, _ = self.lstm(x)                                         # (B, T, H)
    attn_w = torch.softmax(self.attn(output).squeeze(-1), dim=1)     # (B, T)
    context = (attn_w.unsqueeze(-1) * output).sum(dim=1)             # (B, H)
    return self.head(self.dropout(context))
```

**Validation**: `python -m pytest tests/unit/ -q` — existing LSTM unit tests must pass with the new readout.

> **⚠ Breaking checkpoint change**: The new `attn` layer adds keys `"attn.weight"` and
> `"attn.bias"` to the model `state_dict`. Any previously exported artifact is missing these
> keys; `load_state_dict()` (called with `strict=True`) will raise a `RuntimeError` at
> runtime startup. Immediately after confirming tests pass, invalidate stale registry
> entries:
>
> ```bash
> echo '{"models": []}' > models/registry.json
> ```
>
> Existing files under `models/checkpoints/` and `models/exported/` can be archived or
> deleted — they are architecturally incompatible with the updated model.

---

### Step 1.7 — Add Mirror Flip Augmentation

**File**: `src/gesture_control/training/augmentation.py`

**1.7.a** Add field to `AugmentationConfig`:

```python
mirror_flip_prob: float = 0.5
```

**1.7.b** Define named layout constants and derive `_X_COORD_DIMS` from them at module
level. Using named constants makes the list self-documenting and prevents silent breakage
when Phase 2 adds joint angles at dims 134–143 (rotation-invariant scalars that require no
x-flip — the derived list naturally excludes them without any manual update):

```python
# Feature layout constants — must stay in sync with LandmarkFeatureTransformer output.
_N_LANDMARKS: int = 21
_POS_START: int = 0
_POS_END: int = _N_LANDMARKS * 3                    # 63 — end of position block
_VEL_START: int = _POS_END                          # 63 — start of velocity block
_VEL_END: int = _VEL_START + _N_LANDMARKS * 3       # 126 — end of velocity block
_SPREAD_DIMS: int = 5                               # dims 126–130, Euclidean distances
_WRIST_TRAJ_X_DIM: int = _VEL_END + _SPREAD_DIMS   # dim 131 — wrist trajectory x-coord

# x-coordinate dims that must be negated during mirror flip.
# Spread (126–130): sign-invariant distances — excluded.
# Joint angles (Phase 2, dims 134–143): rotation-invariant scalars — excluded.
_X_COORD_DIMS: list[int] = (
    list(range(_POS_START, _POS_END, 3))    # position x: 0, 3, 6, ..., 60
    + list(range(_VEL_START, _VEL_END, 3))  # velocity x: 63, 66, ..., 123
    + [_WRIST_TRAJ_X_DIM]                   # wrist trajectory x: 131
)
```

**1.7.c** Add `_mirror_flip` method to `GestureAugmenter`:

```python
def _mirror_flip(self, seq: np.ndarray) -> np.ndarray:
    """Negate all x-coordinates to simulate a mirrored hand."""
    result = seq.copy()
    result[:, _X_COORD_DIMS] *= -1
    return result
```

**1.7.d** Add the flip call in `augment()` after the existing spatial jitter block:

```python
if self._rng.random() < self._config.mirror_flip_prob:
    result = self._mirror_flip(result)
```

**1.7.e** Update `_POSITION_DIMS` to reflect the new layout:

```python
_POSITION_DIMS: int = 63          # unchanged; spread and wrist_traj handled separately
```

**1.7.f** Expose `mirror_flip_prob` in `_parse_augmentation_config()` in `train_lstm.py`:

```python
mirror_flip_prob=float(aug_data.get("mirror_flip_prob", 0.5)),
```

**1.7.g** Add to `configs/model/lstm_default.yaml` under `augmentation`:

```yaml
augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2]
  spatial_jitter_sigma: 0.005
  apply_prob: 0.5
  mirror_flip_prob: 0.5 # add this line
```

**Validation**: `python -m pytest tests/unit/ -q`

> **Test fixture update required**: `tests/unit/test_augmentation.py` uses
> `_make_seq(T=24, F=126)` throughout. After this step the feature dimension is 134.
> Update the default value of the `F` parameter in `_make_seq()` to `134` and verify that
> the `test_spatial_jitter_only_affects_position_dims` test (which asserts dims beyond
> `_POSITION_DIMS` are unchanged by jitter) still passes — it remains correct because only
> dims 0–62 receive noise regardless of total feature width.

---

### Step 1.8 — Add Label Smoothing

**File**: `src/gesture_control/training/train_lstm.py`

**Prerequisite**: Fix R4 must be applied first (introduces `_build_criterion()`).

Update both `_build_criterion()` calls in `train_lstm()` to pass `label_smoothing=0.1`:

```python
# weighted branch:
criterion: nn.Module = _build_criterion(class_weights, label_smoothing=0.1)
# unweighted branch:
criterion = _build_criterion(None, label_smoothing=0.1)
```

Because Fix R4 consolidated both `nn.CrossEntropyLoss` constructions into one helper, this
is a two-call-site change with no risk of missing a third occurrence.

---

### Step 1.9 — Full Quality Gate

Run before training:

```bash
conda activate gesture-control
python -m ruff check .
python -m mypy src
python -m pytest tests/ -m "not training" -q
```

All must report zero errors/failures.

Additionally run the following two spot-checks before proceeding:

```bash
# 1. Confirm _zero_vector() returns exactly 134 dimensions.
python -c "
from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer
t = LandmarkFeatureTransformer()
assert len(t._zero_vector()) == 134, f'Expected 134, got {len(t._zero_vector())}'
print('Feature dim OK: 134')
"

# 2. Confirm the rebuilt dataset embeds the correct schema version and orientation flag.
python - <<'EOF'
import numpy as np, json
d = np.load("data/processed/ipn_dataset.npz", allow_pickle=True)
meta = json.loads(str(d["metadata_json"].item()))
assert meta["feature_schema_version"] == "feature-schema-v2", (
    f"Expected feature-schema-v2, got {meta['feature_schema_version']}"
)
assert meta["feature_dim"] == 134, f"Expected 134, got {meta['feature_dim']}"
assert meta["enable_orientation_normalization"] is True
print("Dataset schema OK:", meta["feature_schema_version"], "dim:", meta["feature_dim"])
EOF
```

Do not proceed to training if either check fails.

---

### Step 1.10 — Train Phase 1 Model

```bash
python scripts/train_model.py \
  --config configs/model/lstm_default.yaml \
  --dataset data/processed/gesture_dataset_v2.npz
```

**Success criteria** (read from the run's `training_metrics.json`):

| Metric            | Required |
| ----------------- | -------- |
| Val accuracy      | ≥ 85%    |
| Val macro F1      | ≥ 84%    |
| zoom_in recall    | ≥ 80%    |
| zoom_out recall   | ≥ 80%    |
| swipe_left recall | ≥ 82%    |

If **any criterion is not met**, do not proceed to Phase 2. Diagnose via the per-class confusion matrix in `training_metrics.json`.

---

## Phase 2 — Architecture Upgrade

**Prerequisite**: Phase 1 success criteria confirmed.  
**Target**: val accuracy ≥ 88–90%

### Step 2.0 — Extend `checkpoint_loader.py` for Multi-Architecture Dispatch _(blocking)_

**Files**: `src/gesture_control/infrastructure/model/checkpoint_loader.py` and
`src/gesture_control/training/export_checkpoint.py`

`load_inference_artifact()` currently hardcodes `GestureLSTMClassifier`. Attempting to load
a TCN artifact without this change applies TCN weights to an LSTM graph, raising a
`RuntimeError` for unexpected/missing `state_dict` keys at runtime startup. This step must
be completed before any TCN model is trained or exported.

**2.0.a** Add a private factory function near the top of `checkpoint_loader.py`, after the
imports:

```python
def _build_model_from_artifact(
    model_config: dict,
    artifact_type: str,
) -> torch.nn.Module:
    """Dispatch model construction based on stored model_name or artifact_type."""
    from gesture_control.infrastructure.model.lstm_model import GestureLSTMClassifier
    from gesture_control.infrastructure.model.tcn_model import GestureTCNClassifier

    model_name = str(model_config.get("model_name", ""))
    is_tcn = artifact_type == "tcn_inference" or model_name.startswith("tcn")
    if is_tcn:
        return GestureTCNClassifier(
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_classes=int(model_config["num_classes"]),
            dropout=float(model_config["dropout"]),
        )
    return GestureLSTMClassifier(
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config["num_classes"]),
    )
```

**2.0.b** Replace the hardcoded `GestureLSTMClassifier(...)` construction inside
`load_inference_artifact()` with:

```python
artifact_type = str(payload.get("artifact_type", "lstm_inference"))
model = _build_model_from_artifact(model_config, artifact_type).to(device)
model.load_state_dict(state_dict)
model.eval()
```

**2.0.c** Update `export_checkpoint.py` to set the `artifact_type` field correctly for TCN
models (Fix R5 already stores `model_name`; this step reads it):

```python
is_tcn = config.model.model_name.startswith("tcn")
artifact_type = "tcn_inference" if is_tcn else "lstm_inference"
payload: dict[str, Any] = {
    "artifact_type": artifact_type,
    ...
}
```

**Validation**: `python -m pytest tests/unit/test_checkpoint_loader.py -q` — confirm all
existing loader tests still pass (they test LSTM loading; the factory falls back to LSTM for
unrecognised model names). Then verify a Phase 1 LSTM artifact still loads cleanly:

```bash
python -c "
from gesture_control.infrastructure.model.checkpoint_loader import load_inference_artifact
print('Loader import OK')
"
```

---

### Step 2.1 — Implement `GestureTCNClassifier`

**New file**: `src/gesture_control/infrastructure/model/tcn_model.py`

Implement a residual dilated TCN with self-attention:

```python
from __future__ import annotations

import torch
from torch import nn


class _ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (3 - 1)
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      padding=pad, dilation=dilation)
        )
        self.chomp = _Chomp1d(pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.chomp(self.conv(x))))
        return self.relu(out + self.residual(x))


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class GestureTCNClassifier(nn.Module):
    """Residual dilated TCN with 2-head self-attention for gesture classification."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dilations = [1, 2, 4, 8]  # receptive field = 31 > window 24
        channels = [input_size] + [hidden_size] * len(dilations)
        self.tcn = nn.Sequential(
            *[
                _ResidualTCNBlock(channels[i], channels[i + 1], d)
                for i, d in enumerate(dilations)
            ]
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=2, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for Conv1d
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, H)
        attn_out, _ = self.attention(tcn_out, tcn_out, tcn_out)
        context = attn_out.mean(dim=1)                          # (B, H)
        return self.head(self.dropout(context))
```

**Validation**: `python -m pytest tests/unit/ -q` and verify no import errors.

---

### Step 2.2 — Add Joint Angle Features (feature dims 134 → 144)

**File**: `src/gesture_control/pipeline/feature_engineering.py`

**2.2.a** Add finger joint index triplets at module level:

```python
# (landmark_a, vertex, landmark_b) for MCP and PIP of each finger.
_JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (0, 1, 2),   # thumb CMC
    (1, 2, 3),   # thumb MCP
    (0, 5, 6),   # index MCP
    (5, 6, 7),   # index PIP
    (0, 9, 10),  # middle MCP
    (9, 10, 11), # middle PIP
    (0, 13, 14), # ring MCP
    (13, 14, 15),# ring PIP
    (0, 17, 18), # pinky MCP
    (17, 18, 19),# pinky PIP
]
```

**2.2.b** Add `_joint_angles` method to `LandmarkFeatureTransformer`:

```python
def _joint_angles(self, normalized: list[list[float]]) -> list[float]:
    angles: list[float] = []
    for a_idx, v_idx, b_idx in _JOINT_TRIPLETS:
        a = self._subtract(normalized[a_idx], normalized[v_idx])
        b = self._subtract(normalized[b_idx], normalized[v_idx])
        norm_a = math.sqrt(sum(c * c for c in a))
        norm_b = math.sqrt(sum(c * c for c in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            angles.append(0.0)
        else:
            cos_theta = max(-1.0, min(1.0, self._dot(a, b) / (norm_a * norm_b)))
            angles.append(math.acos(cos_theta))
    return angles
```

**2.2.c** Append joint angles to `vector` in `transform()`:

```python
angles = self._joint_angles(normalized)
vector = (
    self._flatten(normalized)
    + self._flatten(velocity)
    + spread
    + wrist_traj
    + angles
)
```

**2.2.d** Update `_zero_vector()`:

```python
return [0.0] * 144
```

> Feature layout after Step 2.2:
>
> - dims 0–62: position (63)
> - dims 63–125: velocity (63)
> - dims 126–130: fingertip spread (5)
> - dims 131–133: wrist trajectory (3)
> - dims 134–143: joint angles (10)
> - **Total: 144 dims**

**2.2.e** Update `_X_COORD_DIMS` in `augmentation.py` — joint angles are rotation-invariant scalars, no x-flip needed; existing list remains valid.

---

### Step 2.3 — Create TCN Training Config

**New file**: `configs/model/tcn_default.yaml`

```yaml
model:
  model_name: tcn_v1
  input_size: 144
  hidden_size: 128
  num_layers: 4 # number of TCN dilation blocks
  dropout: 0.1
  num_classes: 8
  feature_schema_version: feature-schema-v3

training:
  epochs: 40
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0001
  seed: 42
  num_workers: 0
  class_weighting: true

splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratified: true

scheduler:
  enabled: true
  factor: 0.5
  patience: 4
  min_lr: 0.000001

export:
  checkpoint_dir: models/checkpoints
  exported_dir: models/exported
  registry_path: models/registry.json
  feature_schema_version: feature-schema-v3

augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2]
  spatial_jitter_sigma: 0.005
  apply_prob: 0.5
  mirror_flip_prob: 0.5
```

---

### Step 2.4 — Update `train_lstm.py` to Dispatch on Model Name

**File**: `src/gesture_control/training/train_lstm.py`

Add a model factory before the `model = GestureLSTMClassifier(...)` line:

```python
from gesture_control.infrastructure.model.tcn_model import GestureTCNClassifier

def _build_model(config: ModelConfig) -> nn.Module:
    if config.model_name.startswith("tcn"):
        return GestureTCNClassifier(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    return GestureLSTMClassifier(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )
```

Replace `model = GestureLSTMClassifier(...).to(device)` with:

```python
model = _build_model(config.model).to(device)
```

---

### Step 2.5 — Regenerate Dataset for Phase 2

```bash
python scripts/build_dataset.py
```

Verify:

```bash
python - <<'EOF'
import numpy as np, json
d = np.load("data/processed/gesture_dataset_v2.npz", allow_pickle=True)
meta = json.loads(str(d["metadata_json"]))
assert meta["feature_dim"] == 144, f"Expected 144, got {meta['feature_dim']}"
print("Dataset OK:", meta)
EOF
```

---

### Step 2.6 — Full Quality Gate

```bash
python -m ruff check .
python -m mypy src
python -m pytest tests/ -m "not training" -q
```

---

### Step 2.7 — Train Phase 2 Model

```bash
python scripts/train_model.py \
  --config configs/model/tcn_default.yaml \
  --dataset data/processed/gesture_dataset_v2.npz
```

**Success criteria**:

| Metric            | Required |
| ----------------- | -------- |
| Val accuracy      | ≥ 88%    |
| Val macro F1      | ≥ 87%    |
| zoom_in recall    | ≥ 85%    |
| zoom_out recall   | ≥ 85%    |
| swipe_left recall | ≥ 87%    |

---

## Dependency Graph

```
Pre-flight R1  dataset_builder.py   save_npz() param  ─┐
Pre-flight R4  train_lstm.py        _build_criterion() ─┼──► Phase 1 starts
Pre-flight R5  export_checkpoint.py model_name field   ─┘

Step 1.1  feature_engineering.py  ─┐
Step 1.2  feature_pipeline.yaml    ├──► Step 1.5  build_dataset.py
Step 1.3  lstm_default.yaml       ─┘          (regen, dim=134, schema-v2)

Step 1.4  dataset_builder.py ──────────► Step 1.5  (purity filter applied at build time)

Steps 1.6, 1.7, 1.8 are independent of each other
(lstm_model.py, augmentation.py, train_lstm.py)
Step 1.6 requires models/registry.json reset immediately after.
All must complete before Step 1.9 quality gate.

                    ▼
          Step 1.9  quality gate (ruff + mypy + pytest + dim + schema assertions)
                    ▼
             Step 1.10  train (dim=134, LSTM+attn)
                    ▼
           Phase 1 gate: val ≥ 85%
                    ▼
Step 2.0  checkpoint_loader.py + export_checkpoint.py  ← blocking for Phase 2
                    ▼
Step 2.1  tcn_model.py             (new file)
Step 2.2  feature_engineering.py  +joint angles → dim=144
Step 2.3  tcn_default.yaml        (new file)
Step 2.4  train_lstm.py           _build_model() dispatch
                    ▼
             Step 2.5  build_dataset.py (regen, dim=144, schema-v3)
                    ▼
             Step 2.6  quality gate
                    ▼
             Step 2.7  train (dim=144, TCN+attn)
                    ▼
           Phase 2 gate: val ≥ 88%
```

---

## Change Summary per File

| File                                                            | Phase      | Change                                                                      |
| --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------- |
| `src/gesture_control/training/dataset_builder.py`               | Pre-flight | R1: `save_npz()` `feature_schema_version` param; Step 1.4 purity filter     |
| `src/gesture_control/training/train_lstm.py`                    | Pre-flight | R4: `_build_criterion()` helper; Step 1.8 `label_smoothing=0.1`; Step 2.4 dispatch |
| `src/gesture_control/training/export_checkpoint.py`             | Pre-flight | R5: `model_name` in artifact; Step 2.0 `artifact_type` for TCN              |
| `src/gesture_control/pipeline/feature_engineering.py`           | 1 + 2      | Add spread + wrist traj (→134), joint angles (→144)                         |
| `src/gesture_control/training/augmentation.py`                  | 1          | Add mirror flip; derive `_X_COORD_DIMS` from named layout constants         |
| `src/gesture_control/infrastructure/model/lstm_model.py`        | 1          | Replace final-step readout with attention-weighted pooling                  |
| `src/gesture_control/infrastructure/model/checkpoint_loader.py` | 2          | Step 2.0: `_build_model_from_artifact()` multi-arch dispatch factory        |
| `src/gesture_control/infrastructure/model/tcn_model.py`         | 2          | New — `GestureTCNClassifier` (residual dilated TCN + self-attention)        |
| `configs/model/feature_pipeline.yaml`                           | 1          | `orientation_normalization: true`                                           |
| `configs/model/lstm_default.yaml`                               | 1          | `input_size: 134`, schema v2, `mirror_flip_prob: 0.5`                       |
| `configs/model/tcn_default.yaml`                                | 2          | New — TCN training config, `input_size: 144`, schema v3                     |
| `scripts/build_ipn_dataset.py`                                  | 1          | `--feature-schema-version` CLI arg forwarded to `save_npz()`                |
| `tests/unit/test_augmentation.py`                               | 1          | Update `_make_seq()` default from `F=126` to `F=134`                        |

---

**Document**: Gesture Recognition Enhancement — Implementation Plan  
**Suite**: Implementation Documentation (`docs/implementation/`)  
**Last Updated**: April 2026 (revised with pre-execution technical review findings)
