from __future__ import annotations

import numpy as np
import pytest

from gesture_control.training.augmentation import (
    AugmentationConfig,
    GestureAugmenter,
    _POSITION_DIMS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seq(T: int = 24, F: int = 126, value: float | None = None) -> np.ndarray:
    """Return a (T, F) float32 array.  Fills with *value* or an arange."""
    if value is not None:
        return np.full((T, F), value, dtype=np.float32)
    return np.arange(T * F, dtype=np.float32).reshape(T, F)


# ---------------------------------------------------------------------------
# Shape and dtype invariants
# ---------------------------------------------------------------------------

def test_augment_preserves_shape() -> None:
    config = AugmentationConfig(apply_prob=1.0, seed=0)
    aug = GestureAugmenter(config)
    seq = _make_seq()
    result = aug.augment(seq)
    assert result.shape == seq.shape


def test_augment_preserves_dtype() -> None:
    config = AugmentationConfig(apply_prob=1.0, seed=0)
    aug = GestureAugmenter(config)
    seq = _make_seq()
    result = aug.augment(seq)
    assert result.dtype == np.float32


def test_augment_does_not_mutate_input() -> None:
    config = AugmentationConfig(apply_prob=1.0, seed=42)
    aug = GestureAugmenter(config)
    seq = _make_seq(value=1.0)
    original = seq.copy()
    aug.augment(seq)
    np.testing.assert_array_equal(seq, original)


# ---------------------------------------------------------------------------
# apply_prob gating
# ---------------------------------------------------------------------------

def test_apply_prob_zero_returns_identical_array() -> None:
    config = AugmentationConfig(apply_prob=0.0, seed=99)
    aug = GestureAugmenter(config)
    seq = _make_seq()
    result = aug.augment(seq)
    np.testing.assert_array_equal(seq, result)


def test_apply_prob_one_modifies_input() -> None:
    """With apply_prob=1.0 at least one augmentation always fires."""
    config = AugmentationConfig(apply_prob=1.0, seed=7)
    aug = GestureAugmenter(config)
    seq = _make_seq(value=1.0)
    result = aug.augment(seq)
    # Either temporal scale or jitter will have changed the values.
    assert not np.array_equal(seq, result)


# ---------------------------------------------------------------------------
# Spatial jitter — only position dims affected
# ---------------------------------------------------------------------------

def test_spatial_jitter_only_modifies_position_dims() -> None:
    """Velocity features (dims 63-125) must remain unchanged after jitter."""
    # temporal_scale_range == (1.0, 1.0) → factor always 1.0 → identity warp
    config = AugmentationConfig(
        temporal_scale_range=(1.0, 1.0),
        spatial_jitter_sigma=0.1,
        apply_prob=1.0,
        seed=0,
    )
    aug = GestureAugmenter(config)
    seq = np.zeros((24, 126), dtype=np.float32)
    result = aug.augment(seq)

    # Position dims should now differ from zero (noise was added).
    assert not np.allclose(result[:, :_POSITION_DIMS], 0.0)
    # Velocity dims must still be exactly zero.
    np.testing.assert_array_equal(result[:, _POSITION_DIMS:], seq[:, _POSITION_DIMS:])


def test_spatial_jitter_sigma_scales_noise() -> None:
    """Larger sigma → larger mean absolute deviation in position features."""
    small = AugmentationConfig(
        temporal_scale_range=(1.0, 1.0), spatial_jitter_sigma=0.001, apply_prob=1.0, seed=1
    )
    large = AugmentationConfig(
        temporal_scale_range=(1.0, 1.0), spatial_jitter_sigma=0.5, apply_prob=1.0, seed=1
    )
    seq = np.zeros((24, 126), dtype=np.float32)
    noise_small = np.abs(GestureAugmenter(small).augment(seq)[:, :_POSITION_DIMS]).mean()
    noise_large = np.abs(GestureAugmenter(large).augment(seq)[:, :_POSITION_DIMS]).mean()
    assert noise_large > noise_small


# ---------------------------------------------------------------------------
# Temporal scaling
# ---------------------------------------------------------------------------

def test_temporal_scale_first_frame_unchanged_for_any_factor() -> None:
    """Output frame 0 always maps to input frame 0 (src_index == 0)."""
    for factor_range in [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]:
        config = AugmentationConfig(
            temporal_scale_range=factor_range,
            spatial_jitter_sigma=0.0,
            apply_prob=1.0,
            seed=0,
        )
        aug = GestureAugmenter(config)
        seq = np.arange(24, dtype=np.float32).reshape(24, 1).repeat(126, axis=1)
        result = aug._temporal_scale(seq)
        assert np.isclose(result[0, 0], 0.0), f"Failed at factor_range={factor_range}"


def test_temporal_scale_slow_interpolates_mid_frame() -> None:
    """factor=0.5: output[1] maps to input[0.5] → value 0.5 (linear interp)."""
    config = AugmentationConfig(
        temporal_scale_range=(0.5, 0.5),
        spatial_jitter_sigma=0.0,
        apply_prob=1.0,
        seed=0,
    )
    aug = GestureAugmenter(config)
    # Values: seq[t, *] == t (easy to reason about interpolation)
    seq = np.arange(24, dtype=np.float32).reshape(24, 1).repeat(126, axis=1)
    result = aug._temporal_scale(seq)
    assert np.isclose(result[1, 0], 0.5), f"Got {result[1, 0]}"


def test_temporal_scale_fast_clamps_to_last_frame() -> None:
    """factor=2.0: output[12] maps to src index 24.0, clamped to 23 → value 23."""
    config = AugmentationConfig(
        temporal_scale_range=(2.0, 2.0),
        spatial_jitter_sigma=0.0,
        apply_prob=1.0,
        seed=0,
    )
    aug = GestureAugmenter(config)
    seq = np.arange(24, dtype=np.float32).reshape(24, 1).repeat(126, axis=1)
    result = aug._temporal_scale(seq)
    assert np.isclose(result[12, 0], 23.0), f"Got {result[12, 0]}"


def test_temporal_scale_identity_is_noop() -> None:
    """factor=1.0 must produce the same values as the input sequence."""
    config = AugmentationConfig(
        temporal_scale_range=(1.0, 1.0),
        spatial_jitter_sigma=0.0,
        apply_prob=1.0,
        seed=0,
    )
    aug = GestureAugmenter(config)
    seq = _make_seq(value=3.0)
    result = aug._temporal_scale(seq)
    np.testing.assert_allclose(result, seq, atol=1e-5)


# ---------------------------------------------------------------------------
# Determinism with fixed seed
# ---------------------------------------------------------------------------

def test_augmenter_is_deterministic_with_fixed_seed() -> None:
    """Two augmenters with the same seed must produce identical results."""
    seq = np.random.default_rng(0).random((24, 126)).astype(np.float32)
    config = AugmentationConfig(apply_prob=0.8, seed=123)
    r1 = GestureAugmenter(config).augment(seq)
    r2 = GestureAugmenter(config).augment(seq)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 2, 4, 24, 48])
def test_temporal_scale_various_lengths(T: int) -> None:
    """Temporal scaling must not crash for any realistic window length."""
    config = AugmentationConfig(apply_prob=1.0, seed=0)
    aug = GestureAugmenter(config)
    seq = np.ones((T, 126), dtype=np.float32)
    result = aug._temporal_scale(seq)
    assert result.shape == (T, 126)
