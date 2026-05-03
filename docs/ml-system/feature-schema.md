# Feature Schema

**Version string**: `feature-schema-v5`  
**Total dimensionality**: 147  
**Sequence format**: `(24, 147)` — 24-frame sliding window at 30 FPS ≈ 800 ms

---

## Feature Blocks

| Block              | Slice       | Dims | Description                                                                |
| ------------------ | ----------- | ---- | -------------------------------------------------------------------------- |
| `position`         | `[0:63]`    | 63   | 21 wrist-relative, scale-normalised landmark positions (x, y, z per joint) |
| `fingertip_spread` | `[63:68]`   | 5    | 5 inter-fingertip Euclidean distances                                      |
| `wrist_trajectory` | `[68:71]`   | 3    | Net wrist displacement from the oldest frame in the current window         |
| `velocity`         | `[71:134]`  | 63   | 21 per-landmark wrist-relative velocity vectors (Δposition per unit time)  |
| `joint_angles`     | `[134:144]` | 10   | 10 MCP + PIP joint angles in radians                                       |
| `wrist_vel_raw`    | `[144:147]` | 3    | Camera-normalised wrist velocity (x, y, z) — primary directional signal    |

Total: 63 + 5 + 3 + 63 + 10 + 3 = **147**

---

## Preprocessing Pipeline

All features are computed by `LandmarkFeatureTransformer` from a `LandmarkPacket` produced by MediaPipe Hands (21 landmarks × 3 coordinates).

### Step 1 — Wrist-relative centering

All 21 landmarks are translated so the wrist (landmark 0) is at the origin:

$$l_i' = l_i - l_0$$

### Step 2 — Scale normalisation

Landmarks are divided by the Euclidean distance from the origin to landmark 9 (middle finger MCP joint), which approximates hand size:

$$l_i'' = \frac{l_i'}{||l_9'||}$$

### Step 3 — Positions block (`[0:63]`)

Flattened wrist-relative, scale-normalised coordinates for all 21 landmarks:
`[x₀, y₀, z₀, x₁, y₁, z₁, …, x₂₀, y₂₀, z₂₀]`

### Step 4 — Fingertip spread (`[63:68]`)

5 Euclidean distances between adjacent fingertip pairs in normalised space. Captures open/closed hand shape independently of wrist pose.

### Step 5 — Wrist trajectory (`[68:71]`)

Net wrist displacement vector `[Δwx, Δwy, Δwz]` between the current frame and the oldest frame in the sliding window buffer. Provides a coarse directional signal for swipe gestures even before sequence classification.

### Step 6 — Velocity block (`[71:134]`)

Per-landmark velocity computed as the difference between consecutive wrist-relative landmark positions divided by the inter-frame time delta:

$$v_i = \frac{l_i^{(t)} - l_i^{(t-1)}}{\Delta t}$$

### Step 7 — Joint angles (`[134:144]`)

10 joint angles (radians): 5 MCP angles + 5 PIP angles, computed from adjacent landmark pairs using the cosine rule. Encodes hand shape independently of scale and position.

### Step 8 — Wrist velocity raw (`[144:147]`)

Camera-normalised wrist velocity `[vwx, vwy, vwz]` computed from consecutive raw (non-wrist-relative) wrist positions. Key directional signal for all swipe classes.

---

## Orientation Normalisation

Orientation normalisation (`orientation_normalization: false` in `feature_pipeline.yaml`) is **disabled** in production. Enabling it rotates landmarks to a canonical wrist-to-middle-finger axis and was found to destroy the velocity coherence required for swipe direction discrimination (see `docs/archive/research/tcn-quality-gap-investigation.md`).

---

## Invalid Frame Handling

When MediaPipe fails to detect a hand (`LandmarkPacket.valid == False`), `LandmarkFeatureTransformer` emits a zero vector with `valid=False`. The `SlidingWindowSequenceBuffer` tracks the ratio of valid frames and rejects windows below `min_valid_ratio: 0.8`.

---

## Schema Versioning

The version string `feature-schema-v5` is embedded in every NPZ dataset artifact and every exported `.pt` inference artifact. Runtime compatibility is enforced by `validate_runtime_compatibility()` in `checkpoint_loader.py`.

---

## Two-Stream Feature Partition (Enhanced Architecture)

The `EnhancedTwoStreamLSTM` (`configs/model/two_stream_attn.yaml`) partitions the 147-dim vector into two semantically coherent streams before recurrent processing. The partition is a deterministic index gather — no additional preprocessing is required.

| Stream                    | Slice(s)                 | Dims | Semantic role                                                                                      |
| ------------------------- | ------------------------ | ---- | -------------------------------------------------------------------------------------------------- |
| **A — Pose / Shape**      | `[0:63]` + `[134:144]`   | 73   | Static hand configuration: wrist-relative landmark positions + MCP/PIP joint angles                |
| **B — Motion / Dynamics** | `[63:134]` + `[144:147]` | 74   | Dynamic hand motion: fingertip spread, wrist trajectory, per-landmark velocity, raw wrist velocity |

**Rationale.** In the single-stream baseline, the 6 directional dims (`wrist_traj[68:71]` and `wrist_vel_raw[144:147]`) compete for gradient weight with the 141 remaining dims inside every LSTM timestep. Multi-head self-attention resolves temporal competition (which timesteps matter) but cannot resolve within-timestep feature competition (which feature dims matter). Stream B gives the critical motion signals a fully independent pathway, eliminating this competition and raising the effective capacity dedicated to swipe direction discrimination from ~4% to ~50% of total model capacity.

**Implementation** (`src/maestro/infrastructure/model/two_stream_attn_lstm.py` — `EnhancedTwoStreamLSTM._split`):

```python
@staticmethod
def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition (B, T, 147) → pose (B, T, 73) and motion (B, T, 74)."""
    pose   = torch.cat([x[..., :63],   x[..., 134:144]], dim=-1)  # (B, T, 73)
    motion = torch.cat([x[..., 63:134], x[..., 144:]],   dim=-1)  # (B, T, 74)
    return pose, motion
```
