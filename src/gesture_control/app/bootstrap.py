from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from gesture_control.evaluation.benchmark_latency import LatencyBenchmark
from gesture_control.infrastructure.camera.opencv_source import OpenCVFrameSource
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.powerpoint_adapter import (
    PowerPointAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.zoom_adapter import (
    ZoomAdapter,
)
from gesture_control.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)
from gesture_control.infrastructure.feedback.overlay_renderer import OverlayRenderer
from gesture_control.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)
from gesture_control.infrastructure.model.checkpoint_loader import (
    InferenceArtifact,
    load_inference_artifact,
    load_latest_inference_artifact,
    validate_runtime_compatibility,
)
from gesture_control.interfaces.feature_transformer import IFeatureTransformer
from gesture_control.interfaces.frame_source import IFrameSource
from gesture_control.interfaces.landmark_extractor import IHandLandmarkExtractor
from gesture_control.interfaces.sequence_buffer import ISequenceBuffer
from gesture_control.pipeline.activation import GestureActivationController
from gesture_control.pipeline.buffering import SlidingWindowSequenceBuffer
from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer
from gesture_control.pipeline.stabilization import PredictionStabilizer
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
    model_artifact: InferenceArtifact
    stabilizer: PredictionStabilizer
    activation_controller: GestureActivationController
    dispatcher: CommandDispatcher
    overlay_renderer: OverlayRenderer
    benchmark: LatencyBenchmark
    show_window: bool


def build_container(
    config_path: str,
    *,
    model_artifact_path: str | None = None,
    registry_path: str = "models/registry.json",
    show_window: bool = True,
) -> AppContainer:
    config_root = Path(config_path)
    config = load_runtime_config(config_root)
    thresholds = _load_thresholds(config_root / "thresholds" / "production_ipn.yaml")
    vocabulary = _load_vocabulary(config_root / "gestures" / "vocabulary_ipn.yaml")
    command_map = _load_command_map(config_root / "gestures" / "command_map_ipn.yaml")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_artifact_path:
        artifact = load_inference_artifact(
            artifact_path=model_artifact_path, device=device
        )
    else:
        artifact = load_latest_inference_artifact(
            registry_path=registry_path, device=device
        )

    validate_runtime_compatibility(runtime_config=config, artifact=artifact)

    stabilizer = PredictionStabilizer(
        default_threshold=float(thresholds.get("acceptance_default", 0.85)),
        class_thresholds=thresholds.get("class_thresholds", {}),
        vote_window=3,
        class_vote_windows=thresholds.get("class_vote_windows") or None,
        ema_alpha=0.4,
        unknown_label="unknown",
    )
    activation_controller = GestureActivationController(
        activation_gesture=vocabulary.get("activation_gesture", "open_palm_hold"),
        command_map=command_map,
        activation_hold_ms=config.inference.activation_hold_ms,
        idle_timeout_ms=config.inference.idle_timeout_ms,
        command_cooldown_ms=config.inference.command_cooldown_ms,
    )
    dispatcher = _build_dispatcher(
        active_app=config.app.active_app,
        allowed_commands=frozenset(command_map.values()),
    )
    overlay_renderer = OverlayRenderer(mirror_display=config.app.mirror_display)
    benchmark = LatencyBenchmark()

    return AppContainer(
        config=config,
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
        model_artifact=artifact,
        stabilizer=stabilizer,
        activation_controller=activation_controller,
        dispatcher=dispatcher,
        overlay_renderer=overlay_renderer,
        benchmark=benchmark,
        show_window=show_window,
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


def _load_thresholds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "acceptance_default": 0.85,
            "class_thresholds": {},
        }

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {
            "acceptance_default": 0.85,
            "class_thresholds": {},
        }

    acceptance = payload.get("acceptance", {})
    class_thresholds = payload.get("class_thresholds", {})
    if not isinstance(acceptance, dict):
        acceptance = {}
    if not isinstance(class_thresholds, dict):
        class_thresholds = {}

    class_vote_windows = payload.get("class_vote_windows", {})
    if not isinstance(class_vote_windows, dict):
        class_vote_windows = {}

    return {
        "acceptance_default": float(acceptance.get("default", 0.85)),
        "class_thresholds": {str(k): float(v) for k, v in class_thresholds.items()},
        "class_vote_windows": {str(k): int(v) for k, v in class_vote_windows.items()},
    }


def _load_vocabulary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "activation_gesture": "open_palm_hold",
        }

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {
            "activation_gesture": "open_palm_hold",
        }

    activation = payload.get("activation_gesture", "open_palm_hold")
    return {
        "activation_gesture": str(activation),
    }


def _load_command_map(path: Path) -> dict[str, str]:
    """Load gesture→command_id mapping from YAML; fall back to IPN defaults."""
    _defaults: dict[str, str] = {
        "swipe_right": "next_slide",
        "swipe_left": "previous_slide",
        "swipe_up": "raise_hand",
        "swipe_down": "start_presentation",
        "zoom_in": "toggle_pointer_mode",
        "zoom_out": "exit_presentation",
    }
    if not path.exists():
        return _defaults
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return _defaults
    mapping = payload.get("gesture_commands", {})
    if not isinstance(mapping, dict):
        return _defaults
    return {str(k): str(v) for k, v in mapping.items() if k and v}


def _build_dispatcher(
    active_app: str,
    allowed_commands: frozenset[str],
) -> CommandDispatcher:
    """Construct the correct adapter for *active_app* and wrap it in a CommandDispatcher."""
    keyboard = KeyboardAdapter()
    app_lower = (active_app or "").strip().lower()
    if app_lower in ("powerpoint", "ppt"):
        adapter = PowerPointAdapter(keyboard=keyboard)
    elif app_lower in ("zoom",):
        adapter = ZoomAdapter(keyboard=keyboard)
    else:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "Unknown active_app=%r — defaulting to PowerPointAdapter", active_app
        )
        adapter = PowerPointAdapter(keyboard=keyboard)
    return CommandDispatcher(
        adapter=adapter,
        allowed_commands=allowed_commands,
    )
