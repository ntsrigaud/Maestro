from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Feature layout constants for the 147-dim vector produced by
# LandmarkFeatureTransformer (feature-schema-v5).
#
#   [_POS_START  : _POS_END]          21 wrist-relative, scale-normalised positions
#   [_POS_END    : _SPREAD_END]       5  fingertip spread (Euclidean distances)
#   [_SPREAD_END : _WRIST_TRAJ_END]   3  wrist trajectory (net displacement)
#   [_VEL_START  : _VEL_END]          21 time-normalised velocity vectors (wrist-relative)
#   [_VEL_END    : _ANGLES_END]       10 joint angles
#   [_ANGLES_END : _WRIST_VEL_END]    3  wrist velocity in camera-normalized coords (v5)
# ---------------------------------------------------------------------------
_N_LANDMARKS: int = 21
_POS_START: int = 0
_POS_END: int = _N_LANDMARKS * 3  # 63
_SPREAD_END: int = _POS_END + 5  # 68  (5 fingertip distances)
_WRIST_TRAJ_END: int = _SPREAD_END + 3  # 71  (3 wrist-trajectory dims)
_VEL_START: int = _WRIST_TRAJ_END  # 71
_VEL_END: int = _VEL_START + _N_LANDMARKS * 3  # 134
_ANGLES_END: int = _VEL_END + 10  # 144
_WRIST_VEL_START: int = _ANGLES_END  # 144
_WRIST_VEL_END: int = _WRIST_VEL_START + 3  # 147  (wrist_vel_raw x/y/z)

# Convenience alias kept for backward-compat with tests that imported it.
_POSITION_DIMS: int = _POS_END  # 63

# X-coordinate indices to negate when performing a horizontal mirror flip.
# Positions x:      every 3rd dim from _POS_START (landmark x-coords).
# Wrist traj x:     the first dim of the wrist-trajectory block (_SPREAD_END).
# Velocity x:       every 3rd dim from _VEL_START (wrist-relative velocity x-coords).
# Wrist vel raw x:  the first dim of the camera-space wrist velocity block (_WRIST_VEL_START).
#                   Negating this ensures a mirrored swipe_right is re-signed as swipe_left.
_X_COORD_DIMS: list[int] = (
    list(range(_POS_START, _POS_END, 3))  # 21 position x-coords
    + [_SPREAD_END]  # 1  wrist-traj x
    + list(range(_VEL_START, _VEL_END, 3))  # 21 velocity x-coords (wrist-relative)
    + [_WRIST_VEL_START]  # 1  wrist velocity raw x (camera space)
)


@dataclass(slots=True)
class AugmentationConfig:
    """Hyper-parameters for on-the-fly gesture sequence augmentation.

    All augmentations are applied independently per sample with probability
    *apply_prob*, so either, both, or neither may fire on any given pass.
    """

    # Speed factor drawn uniformly from this range on each applicable sample.
    # factor < 1 → slower gesture (only the beginning of the sequence is visible
    #               in the T-frame window; the rest stays at the last value).
    # factor > 1 → faster gesture (the sequence is compressed; later frames
    #               are clamped to the final input frame).
    temporal_scale_range: tuple[float, float] = (0.8, 1.2)

    # Standard deviation of Gaussian noise added to normalised position features.
    # With hand size ≈ 1 normalised unit and positions roughly in [-2, 2],
    # σ = 0.005 is sub-pixel and semantics-preserving.
    spatial_jitter_sigma: float = 0.005

    # Independent probability of applying each augmentation to a given sample.
    apply_prob: float = 0.5

    # Optional fixed seed for reproducible augmentation (useful in unit tests).
    seed: int | None = None

    # Probability of applying horizontal mirror flip per sample.
    # Negates all x-coordinate dims (positions, wrist-traj x, velocities x)
    # to simulate the opposite handedness / viewing angle.
    mirror_flip_prob: float = 0.5


class GestureAugmenter:
    """Applies temporal-speed, spatial-jitter, and mirror-flip augmentation.

    Each call to :meth:`augment` returns a new array; the input is never mutated.
    All augmentations are applied independently with probability *apply_prob*
    (mirror flip uses *mirror_flip_prob*).
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """Return an augmented copy of *sequence* (shape ``(T, F)``).

        The input array is never mutated.  Augmentations are applied in order:
        temporal scale → spatial jitter → mirror flip.
        """
        result = sequence.copy()
        if self._rng.random() < self._config.apply_prob:
            result = self._temporal_scale(result)
        if self._rng.random() < self._config.apply_prob:
            result = self._spatial_jitter(result)
        if self._rng.random() < self._config.mirror_flip_prob:
            result = self._mirror_flip(result)
        return result

    # ------------------------------------------------------------------
    # Private augmentation kernels
    # ------------------------------------------------------------------

    def _temporal_scale(self, seq: np.ndarray) -> np.ndarray:
        """Resample the sequence to simulate gestures at different speeds.

        A factor *f* is sampled uniformly from *temporal_scale_range*:

        * f > 1 — each output frame corresponds to a larger step through the
          original frames, making the gesture appear *faster*.  Once the
          mapped index exceeds ``T – 1`` it is clamped, so the tail of the
          window holds the last gesture frame.
        * f < 1 — each output frame corresponds to a smaller step, making the
          gesture appear *slower*.  Only the first ``⌊T × f⌋`` original frames
          are covered; the tail shows the point where the gesture had reached.

        Window length ``T`` and feature width ``F`` are preserved.  Interpolation
        is linear per-feature via :func:`numpy.interp`.
        """
        T, F = seq.shape
        # rng.uniform(a, a) == a when low == high (numpy contracts to 0-range).
        factor = float(
            self._rng.uniform(
                low=self._config.temporal_scale_range[0],
                high=self._config.temporal_scale_range[1],
            )
        )
        src_indices = np.clip(np.arange(T, dtype=np.float64) * factor, 0.0, float(T - 1))
        # Vectorised linear interpolation across all F features simultaneously.
        # This replaces the original O(F) Python loop (np.interp × F calls) with
        # two gather operations and a single broadcast lerp — ~12× faster.
        idx_low = np.floor(src_indices).astype(np.intp)  # (T,)
        idx_high = np.minimum(idx_low + 1, T - 1)  # (T,) clamped
        frac = (src_indices - idx_low).astype(seq.dtype)[:, np.newaxis]  # (T,1)
        return seq[idx_low] * (1.0 - frac) + seq[idx_high] * frac  # (T, F)

    def _spatial_jitter(self, seq: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to the position features of the sequence.

        Velocity features are intentionally unchanged so that the derived
        motion signal is not independently distorted.
        """
        result = seq.copy()
        noise = self._rng.normal(
            loc=0.0,
            scale=self._config.spatial_jitter_sigma,
            size=(seq.shape[0], _POSITION_DIMS),
        )
        result[:, :_POSITION_DIMS] += noise.astype(seq.dtype)
        return result

    def _mirror_flip(self, seq: np.ndarray) -> np.ndarray:
        """Horizontally mirror the gesture by negating all x-coordinate dims.

        Position x, wrist-trajectory x, and velocity x dims are negated.
        Fingertip spread dims are unsigned distances and are left unchanged.
        """
        result = seq.copy()
        result[:, _X_COORD_DIMS] *= -1
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Graph-space augmentation (graph-schema-v1, shape (T, N, C))
# ──────────────────────────────────────────────────────────────────────────────

# Node index ranges for mirror-flip hand swap.
_RIGHT_HAND_SLICE = slice(0, 21)  # nodes 0–20  (MediaPipe right hand)
_LEFT_HAND_SLICE = slice(21, 42)  # nodes 21–41 (MediaPipe left hand)
_X_COORD_IDX: int = 0  # coordinate dimension for horizontal axis


@dataclass(slots=True)
class GraphAugmentationConfig:
    """Hyper-parameters for on-the-fly graph sequence augmentation.

    All augmentations are applied independently per sample with probability
    *apply_prob*.
    """

    # Speed factor drawn uniformly from this range.
    temporal_scale_range: tuple[float, float] = (0.8, 1.2)
    # Std-dev of Gaussian noise added to every node coordinate.
    spatial_jitter_sigma: float = 0.005
    # Per-augmentation application probability.
    apply_prob: float = 0.5
    # Mirror-flip probability.
    mirror_flip_prob: float = 0.5
    # Optional seed for reproducibility.
    seed: int | None = None


class GraphGestureAugmenter:
    """Augmenter for graph sequences of shape ``(T, N, C)``.

    Augmentations:
    * **Temporal scale** — resample time axis to simulate speed variation.
    * **Spatial jitter** — add Gaussian noise to all node coordinates.
    * **Mirror flip** — negate x-axis and swap right/left hand node blocks.

    Each call to :meth:`augment` returns a new array; the input is never mutated.
    """

    def __init__(self, config: GraphAugmentationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """Return an augmented copy of *sequence* (shape ``(T, N, C)``).

        Augmentations are applied in order:
        temporal scale → spatial jitter → mirror flip.
        """
        result = sequence.copy()
        if self._rng.random() < self._config.apply_prob:
            result = self._temporal_scale(result)
        if self._rng.random() < self._config.apply_prob:
            result = self._spatial_jitter(result)
        if self._rng.random() < self._config.mirror_flip_prob:
            result = self._mirror_flip(result)
        return result

    # ------------------------------------------------------------------
    # Private augmentation kernels
    # ------------------------------------------------------------------

    def _temporal_scale(self, seq: np.ndarray) -> np.ndarray:
        """Resample the time axis to simulate gestures at different speeds."""
        T, N, C = seq.shape
        factor = float(
            self._rng.uniform(
                low=self._config.temporal_scale_range[0],
                high=self._config.temporal_scale_range[1],
            )
        )
        src_indices = np.clip(np.arange(T, dtype=np.float64) * factor, 0.0, float(T - 1))
        orig_indices = np.arange(T, dtype=np.float64)
        result = np.empty_like(seq)
        # Interpolate each (node, coord) channel independently.
        flat = seq.reshape(T, N * C)
        for dim in range(N * C):
            result.reshape(T, N * C)[:, dim] = np.interp(src_indices, orig_indices, flat[:, dim])
        return result

    def _spatial_jitter(self, seq: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to all node coordinates."""
        noise = self._rng.normal(
            loc=0.0,
            scale=self._config.spatial_jitter_sigma,
            size=seq.shape,
        ).astype(seq.dtype)
        return seq + noise

    def _mirror_flip(self, seq: np.ndarray) -> np.ndarray:
        """Mirror the gesture: negate x-axis and swap right/left hand nodes."""
        result = seq.copy()
        # Negate the x-coordinate for all nodes.
        result[:, :, _X_COORD_IDX] *= -1
        # Swap right-hand and left-hand node blocks so the model sees the
        # correct semantic layout for the mirrored handedness.
        rh = result[:, _RIGHT_HAND_SLICE, :].copy()
        result[:, _RIGHT_HAND_SLICE, :] = result[:, _LEFT_HAND_SLICE, :]
        result[:, _LEFT_HAND_SLICE, :] = rh
        return result
