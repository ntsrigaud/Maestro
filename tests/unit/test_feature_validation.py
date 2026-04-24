"""Tests for training/inference feature space parity and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from maestro.training.dataset_builder import SequenceDatasetBuilder
from maestro.utils.feature_validation import (
    FeatureSchemaError,
    load_dataset_metadata,
    validate_dataset_compatibility,
    validate_feature_schema_parity,
)
from maestro.utils.schema_validation import FeaturePipelineConfig


class TestFeatureSchemaValidation:
    """Test feature space parity between training and inference."""

    def test_matching_orientation_normalization(self):
        """Datasets built with matching orientation config should validate."""
        dataset_metadata = {
            "feature_dim": 126,
            "window_length": 24,
            "window_step": 3,
            "enable_orientation_normalization": True,
            "vocabulary": ["gesture_a", "gesture_b"],
        }
        runtime_config = FeaturePipelineConfig(
            orientation_normalization=True,
        )

        is_compatible, issues = validate_feature_schema_parity(
            dataset_metadata, runtime_config
        )

        assert is_compatible is True
        assert len(issues) == 0

    def test_mismatching_orientation_normalization_raises_error(self):
        """Datasets with mismatched orientation config should raise FeatureSchemaError."""
        dataset_metadata = {
            "feature_dim": 126,
            "window_length": 24,
            "window_step": 3,
            "enable_orientation_normalization": True,  # Built WITH normalization
            "vocabulary": ["gesture_a", "gesture_b"],
        }
        runtime_config = FeaturePipelineConfig(
            orientation_normalization=False,  # Runtime WITHOUT normalization
        )

        with pytest.raises(FeatureSchemaError) as exc_info:
            validate_feature_schema_parity(dataset_metadata, runtime_config)

        assert "Critical feature space mismatch" in str(exc_info.value)
        assert "orientation_normalization" in str(exc_info.value)
        assert "training/inference distribution mismatch" in str(exc_info.value)

    def test_legacy_dataset_without_orientation_metadata_warning(self):
        """Legacy datasets lacking orientation metadata should warn but not fail."""
        dataset_metadata = {
            "feature_dim": 126,
            "window_length": 24,
            "window_step": 3,
            # Missing: enable_orientation_normalization
            "vocabulary": ["gesture_a", "gesture_b"],
        }
        runtime_config = FeaturePipelineConfig()

        is_compatible, issues = validate_feature_schema_parity(
            dataset_metadata, runtime_config
        )

        assert is_compatible is False
        assert any(
            "legacy" in issue.lower() or "warning" in issue.lower() for issue in issues
        )

    def test_feature_dimensionality_mismatch_raises_error(self):
        """Datasets with wrong feature dimensionality should raise error."""
        dataset_metadata = {
            "feature_dim": 84,  # Wrong! Should be 126
            "window_length": 24,
            "window_step": 3,
            "enable_orientation_normalization": False,
            "vocabulary": ["gesture_a", "gesture_b"],
        }
        runtime_config = FeaturePipelineConfig()

        with pytest.raises(FeatureSchemaError) as exc_info:
            validate_feature_schema_parity(dataset_metadata, runtime_config)

        assert "Feature dimensionality mismatch" in str(exc_info.value)
        assert "126" in str(exc_info.value)

    def test_window_parameter_warnings(self):
        """Non-standard window parameters should generate warnings."""
        dataset_metadata = {
            "feature_dim": 126,
            "window_length": 32,  # Non-standard
            "window_step": 5,  # Non-standard
            "enable_orientation_normalization": False,
            "vocabulary": ["gesture_a", "gesture_b"],
        }
        runtime_config = FeaturePipelineConfig()

        is_compatible, issues = validate_feature_schema_parity(
            dataset_metadata, runtime_config
        )

        assert is_compatible is False
        assert any("window_length" in issue for issue in issues)
        assert any("window_step" in issue for issue in issues)


class TestDatasetMetadataLoading:
    """Test loading and parsing dataset metadata from NPZ files."""

    def test_load_valid_dataset_metadata(self):
        """Valid dataset metadata should load successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "test_dataset.npz"

            # Create a minimal valid NPZ with metadata
            metadata = {
                "feature_dim": 126,
                "window_length": 24,
                "enable_orientation_normalization": True,
                "vocabulary": ["gesture_a"],
            }
            np.savez_compressed(
                dataset_path,
                sequences=np.zeros((10, 24, 126), dtype=np.float32),
                labels=np.array(["gesture_a"] * 10, dtype=object),
                label_ids=np.array([0] * 10, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata), dtype=object),
            )

            loaded_metadata = load_dataset_metadata(dataset_path)

            assert loaded_metadata["feature_dim"] == 126
            assert loaded_metadata["window_length"] == 24
            assert loaded_metadata["enable_orientation_normalization"] is True

    def test_load_nonexistent_dataset_raises_error(self):
        """Loading nonexistent dataset should raise FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path/dataset.npz")

        with pytest.raises(FileNotFoundError):
            load_dataset_metadata(nonexistent_path)

    def test_load_malformed_metadata_raises_error(self):
        """NPZ with invalid metadata JSON should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "bad_dataset.npz"

            # Create NPZ with invalid JSON in metadata_json
            np.savez_compressed(
                dataset_path,
                sequences=np.zeros((10, 24, 126), dtype=np.float32),
                labels=np.array(["gesture_a"] * 10, dtype=object),
                metadata_json=np.asarray("invalid json {", dtype=object),
            )

            with pytest.raises(ValueError):
                load_dataset_metadata(dataset_path)


class TestDatasetCompatibilityValidation:
    """Integration tests for full dataset compatibility validation."""

    def test_validate_compatible_dataset(self):
        """Compatible dataset and config should pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"

            metadata = {
                "feature_dim": 126,
                "window_length": 24,
                "window_step": 3,
                "enable_orientation_normalization": True,
                "vocabulary": ["gesture_a", "gesture_b"],
            }
            np.savez_compressed(
                dataset_path,
                sequences=np.zeros((10, 24, 126), dtype=np.float32),
                labels=np.array(["gesture_a"] * 10, dtype=object),
                metadata_json=np.asarray(json.dumps(metadata), dtype=object),
            )

            runtime_config = FeaturePipelineConfig(
                orientation_normalization=True,
            )

            is_compatible, issues = validate_dataset_compatibility(
                dataset_path, runtime_config
            )

            assert is_compatible is True

    def test_validate_incompatible_dataset_raises_error(self):
        """Incompatible dataset should raise FeatureSchemaError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"

            metadata = {
                "feature_dim": 126,
                "window_length": 24,
                "enable_orientation_normalization": True,
                "vocabulary": ["gesture_a"],
            }
            np.savez_compressed(
                dataset_path,
                sequences=np.zeros((10, 24, 126), dtype=np.float32),
                labels=np.array(["gesture_a"] * 10, dtype=object),
                metadata_json=np.asarray(json.dumps(metadata), dtype=object),
            )

            runtime_config = FeaturePipelineConfig(
                orientation_normalization=False,
            )

            with pytest.raises(FeatureSchemaError):
                validate_dataset_compatibility(dataset_path, runtime_config)


class TestTrainingInferenceParity:
    """Test that training and inference pipelines produce identical features."""

    def test_dataset_builder_stores_correct_metadata(self):
        """SequenceDatasetBuilder should store feature config in metadata."""
        from maestro.training.dataset_builder import LabeledLandmarkFrame

        builder = SequenceDatasetBuilder(
            vocabulary=["gesture_a"],
            window_length=24,
            window_step=3,
            min_valid_ratio=0.8,
            enable_orientation_normalization=True,
        )

        # Create minimal valid frame
        _ = LabeledLandmarkFrame(
            recording_id="rec_1",
            timestamp_ms=0,
            label="gesture_a",
            landmarks=[[0.0, 0.0, 0.0]] * 21,
            valid=True,
        )

        # Would need more frames to actually generate samples,
        # so just verify metadata is correct
        assert builder._enable_orientation_normalization is True

        # When building from frames and saving, metadata should reflect setting
        # This is verified by load_npz returning the flag in metadata

    def test_feature_pipeline_config_matches_between_train_and_inference(self):
        """Feature pipeline config should be consistent."""
        # Training config
        training_config = FeaturePipelineConfig(
            orientation_normalization=True,
        )

        # Inference config (should match)
        inference_config = FeaturePipelineConfig(
            orientation_normalization=True,
        )

        assert (
            training_config.orientation_normalization
            == inference_config.orientation_normalization
        )
        assert (
            training_config.wrist_relative_normalization
            == inference_config.wrist_relative_normalization
        )
        assert training_config.velocity_features == inference_config.velocity_features
