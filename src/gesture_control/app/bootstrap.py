from __future__ import annotations

from dataclasses import dataclass

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
        enable_orientation_normalization=False,
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
