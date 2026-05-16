"""Feature space validation utilities for training/inference parity assurance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maestro.utils.schema_validation import FeaturePipelineConfig


class FeatureSchemaError(ValueError):
    """Raised when dataset feature schema is incompatible with runtime configuration."""

    pass


def load_dataset_metadata(dataset_path: str | Path) -> dict[str, Any]:
    """Load and validate feature schema metadata from dataset.

    Args:
        dataset_path: Path to the NPZ dataset file

    Returns:
        Metadata dictionary containing feature schema information

    Raises:
        FileNotFoundError: If dataset file not found
        ValueError: If metadata is malformed
    """
    import numpy as np

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        with np.load(path, allow_pickle=True) as data:
            metadata_str = str(data["metadata_json"].item())
            metadata = json.loads(metadata_str)
            return metadata
    except Exception as exc:
        raise ValueError(f"Failed to load dataset metadata: {exc}") from exc


def validate_feature_schema_parity(
    dataset_metadata: dict[str, Any],
    runtime_feature_config: FeaturePipelineConfig,
) -> tuple[bool, list[str]]:
    """Validate that dataset feature schema matches runtime feature configuration.

    This is a critical safety check that prevents training/inference distribution
    mismatch. If the LSTM was trained with orientation normalization enabled,
    inference must also have it enabled.

    Args:
        dataset_metadata: Metadata dictionary from load_dataset_metadata()
        runtime_feature_config: FeaturePipelineConfig from runtime environment

    Returns:
        Tuple of (is_compatible, list_of_issues)
        - is_compatible: True if schema matches, False otherwise
        - list_of_issues: Human-readable descriptions of any mismatches

    Raises:
        FeatureSchemaError: If critical incompatibilities detected
    """
    issues: list[str] = []

    # Check orientation normalization parity (most critical)
    dataset_orientation = dataset_metadata.get("enable_orientation_normalization")
    runtime_orientation = runtime_feature_config.orientation_normalization

    if dataset_orientation is None:
        # Legacy dataset without metadata; warn but allow
        issues.append(
            "WARNING: Dataset lacks orientation_normalization metadata. "
            "Cannot verify feature space parity. Dataset may have been built "
            "with older version. Consider regenerating dataset."
        )
    elif dataset_orientation != runtime_orientation:
        raise FeatureSchemaError(
            f"Critical feature space mismatch: "
            f"Dataset was built with orientation_normalization={dataset_orientation}, "
            f"but runtime is configured with "
            f"orientation_normalization={runtime_orientation}. "
            f"This creates a training/inference distribution mismatch. "
            f"Either: (1) regenerate dataset with matching config, or "
            f"(2) update runtime config to match dataset. "
            f"Detected in: {dataset_metadata}"
        )

    # Check feature dimensionality — allowed dims per schema version.
    # feature-schema-v1: 126 (21×3 positions + 21×3 velocities)
    # feature-schema-v2: 134 (positions + 5 spread + 3 wrist_traj + velocities)
    # feature-schema-v3: 144 (v2 + 10 joint angles)
    # feature-schema-v4: 144 (same dims as v3, orientation_normalization disabled)
    _SCHEMA_DIM: dict[str, int] = {
        "feature-schema-v1": 126,
        "feature-schema-v2": 134,
        "feature-schema-v3": 144,
        "feature-schema-v4": 144,
    }
    dataset_feature_dim = dataset_metadata.get("feature_dim")
    schema_ver = dataset_metadata.get("feature_schema_version", "feature-schema-v1")
    expected_feature_dim = _SCHEMA_DIM.get(schema_ver, 126)

    if dataset_feature_dim is not None and dataset_feature_dim != expected_feature_dim:
        raise FeatureSchemaError(
            f"Feature dimensionality mismatch: "
            f"Dataset has {dataset_feature_dim} dims, "
            f"expected {expected_feature_dim} for schema {schema_ver!r}. "
            f"Dataset may have been built with incompatible feature schema."
        )

    # Check window parameters
    dataset_window_length = dataset_metadata.get("window_length")
    if dataset_window_length != 24:
        issues.append(
            f"WARNING: Dataset window_length={dataset_window_length}, expected 24. "
            f"Inference window_length should match. "
            f"This may affect sequence alignment."
        )

    dataset_window_step = dataset_metadata.get("window_step")
    if dataset_window_step != 3:
        issues.append(
            f"WARNING: Dataset window_step={dataset_window_step}, expected 3. "
            f"This affects sampling rate during inference."
        )

    is_compatible = len(issues) == 0
    return is_compatible, issues


def validate_dataset_compatibility(
    dataset_path: str | Path,
    runtime_feature_config: FeaturePipelineConfig,
) -> tuple[bool, list[str]]:
    """High-level validation that dataset is compatible with runtime.

    This is the recommended entry point for runtime initialization.
    Call this before loading the model and dataset for inference.

    Args:
        dataset_path: Path to NPZ dataset file
        runtime_feature_config: FeaturePipelineConfig from runtime environment

    Returns:
        Tuple of (is_compatible, list_of_issues)

    Raises:
        FeatureSchemaError: If critical incompatibilities detected
        FileNotFoundError: If dataset not found
        ValueError: If metadata malformed
    """
    metadata = load_dataset_metadata(dataset_path)
    return validate_feature_schema_parity(metadata, runtime_feature_config)
