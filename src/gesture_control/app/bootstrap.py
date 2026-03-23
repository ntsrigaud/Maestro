from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gesture_control.infrastructure.camera.opencv_source import OpenCVFrameSource
from gesture_control.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)
from gesture_control.interfaces.feature_transformer import IFeatureTransformer
from gesture_control.interfaces.frame_source import IFrameSource
from gesture_control.interfaces.landmark_extractor import IHandLandmarkExtractor
from gesture_control.interfaces.sequence_buffer import ISequenceBuffer
from gesture_control.pipeline.buffering import SlidingWindowSequenceBuffer
from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer
from gesture_control.utils.config_loader import load_runtime_config
from gesture_control.utils.feature_validation import validate_dataset_compatibility
from gesture_control.utils.schema_validation import RuntimeConfig


@dataclass(slots=True)
class AppContainer:
    config: RuntimeConfig
    frame_source: IFrameSource
    landmark_extractor: IHandLandmarkExtractor
    feature_transformer: IFeatureTransformer
    sequence_buffer: ISequenceBuffer


def build_container(config_path: str) -> AppContainer:
    config = load_runtime_config(config_path)
    frame_source = OpenCVFrameSource(
        camera_index=config.app.camera_index,
        frame_width=config.app.frame_width,
        frame_height=config.app.frame_height,
        target_fps=config.app.target_fps,
    )
    landmark_extractor = MediaPipeHandsExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    feature_transformer = LandmarkFeatureTransformer(
        enable_orientation_normalization=config.feature_pipeline.orientation_normalization,
    )
    sequence_buffer = SlidingWindowSequenceBuffer(
        window_length=config.inference.window_length,
        window_step=config.inference.window_step,
        min_valid_ratio=config.inference.min_valid_ratio,
    )
    return AppContainer(
        config=config,
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
    )


def validate_container_dataset(
    container: AppContainer,
    dataset_path: str | Path,
) -> None:
    """Validate that dataset is compatible with container configuration.

    This is a safety check that must be called before using the dataset
    in training or inference. Raises FeatureSchemaError if incompatibilities
    are detected.

    Args:
        container: AppContainer from build_container()
        dataset_path: Path to NPZ dataset file

    Raises:
        FeatureSchemaError: If feature schema incompatibility detected
        FileNotFoundError: If dataset not found
        ValueError: If metadata malformed
    """
    is_compatible, issues = validate_dataset_compatibility(
        dataset_path, container.config.feature_pipeline
    )
    if not is_compatible:
        for issue in issues:
            print(f"WARNING: {issue}")
    # Note: is_compatible=False with only warnings is acceptable
    # FeatureSchemaError will be raised inside validate_dataset_compatibility
    # if critical incompatibilities are detected
