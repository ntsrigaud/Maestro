from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Feature layout produced by LandmarkFeatureTransformer:
#   dims  0-62  — 21 wrist-relative, scale-normalised landmark positions (x, y, z)
#   dims 63-125 — 21 time-normalised velocity vectors (x, y, z)
# Augmentation noise is applied only to the position dims so the velocity
# structure (already derived from position differences) is not independently
# corrupted.
_POSITION_DIMS: int = 63


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


class GestureAugmenter:
    """Applies temporal-speed and spatial-jitter augmentation to gesture sequences.

    Each call to :meth:`augment` returns a new array; the input is never mutated.
    Both augmentations are applied independently with probability *apply_prob*,
    making it possible for neither, one, or both to trigger on any given sample.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """Return an augmented copy of *sequence* (shape ``(T, F)``).

        The input array is never mutated.  Both augmentations are gated by
        :attr:`AugmentationConfig.apply_prob` and applied in the order:
        temporal scale → spatial jitter.
        """
        result = sequence.copy()
        if self._rng.random() < self._config.apply_prob:
            result = self._temporal_scale(result)
        if self._rng.random() < self._config.apply_prob:
            result = self._spatial_jitter(result)
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
        src_indices = np.clip(
            np.arange(T, dtype=np.float64) * factor, 0.0, float(T - 1)
        )
        orig_indices = np.arange(T, dtype=np.float64)
        result = np.empty_like(seq)
        for dim in range(F):
            result[:, dim] = np.interp(src_indices, orig_indices, seq[:, dim])
        return result

    def _spatial_jitter(self, seq: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to the position features of the sequence.

        Velocity features (dims 63–125) are intentionally unchanged so that
        the derived motion signal is not independently distorted.
        """
        result = seq.copy()
        noise = self._rng.normal(
            loc=0.0,
            scale=self._config.spatial_jitter_sigma,
            size=(seq.shape[0], _POSITION_DIMS),
        )
        result[:, :_POSITION_DIMS] += noise.astype(seq.dtype)
        return result
