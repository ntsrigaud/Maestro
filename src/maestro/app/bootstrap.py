from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from maestro.evaluation.benchmark_latency import LatencyBenchmark
from maestro.infrastructure.camera.opencv_source import OpenCVFrameSource
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from maestro.infrastructure.dispatch.app_adapters.presentation_adapter import (
    PresentationAdapter,
)
from maestro.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)
from maestro.infrastructure.feedback.overlay_renderer import OverlayRenderer
from maestro.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)
from maestro.infrastructure.mediapipe.tasks_hands_extractor import (
    MediaPipeTasksHandsExtractor,
)
from maestro.infrastructure.model.checkpoint_loader import (
    InferenceArtifact,
    load_inference_artifact,
    validate_runtime_compatibility,
)
from maestro.interfaces.command_dispatcher import ICommandDispatcher
from maestro.interfaces.feature_transformer import IFeatureTransformer
from maestro.interfaces.feedback_renderer import IFeedbackRenderer
from maestro.interfaces.frame_source import IFrameSource
from maestro.interfaces.landmark_extractor import IHandLandmarkExtractor
from maestro.interfaces.sequence_buffer import ISequenceBuffer
from maestro.pipeline.activation import GestureActivationController
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer
from maestro.pipeline.hand_selection import HandSelector
from maestro.pipeline.stabilization import PredictionStabilizer
from maestro.utils.config_loader import load_runtime_config
from maestro.utils.feature_validation import validate_dataset_compatibility
from maestro.utils.schema_validation import RuntimeConfig


@dataclass(slots=True)
class AppContainer:
    config: RuntimeConfig
    frame_source: IFrameSource
    landmark_extractor: IHandLandmarkExtractor
    hand_selector: HandSelector
    feature_transformer: IFeatureTransformer
    sequence_buffer: ISequenceBuffer
    model_artifact: InferenceArtifact
    stabilizer: PredictionStabilizer
    activation_controller: GestureActivationController
    dispatcher: ICommandDispatcher
    overlay_renderer: IFeedbackRenderer
    benchmark: LatencyBenchmark
    show_window: bool
    # Model output → display/dispatch label remapping (from command map YAML).
    # Applied after stabilization: replaces the model's raw label with its
    # user-facing equivalent before command dispatch and UI rendering.
    label_remap: dict[str, str] = field(default_factory=dict)


def build_container(
    config_path: str,
    *,
    model_artifact_path: str | None = None,
    hf_revision_override: str | None = None,
    show_window: bool = False,
    minimal_preview: bool = False,
) -> AppContainer:
    config_root = Path(config_path)
    config = load_runtime_config(config_root)
    thresholds = _load_thresholds(config_root / "thresholds" / "production_hybrid.yaml")
    command_map, continuous_labels, label_remap = _load_command_map(
        config_root / "gestures" / config.app.command_map
    )

    frame_source = OpenCVFrameSource(
        camera_index=config.app.camera_index,
        frame_width=config.app.frame_width,
        frame_height=config.app.frame_height,
        target_fps=config.app.target_fps,
    )

    device = _select_device()
    if model_artifact_path:
        artifact = load_inference_artifact(artifact_path=model_artifact_path, device=device)
    else:
        artifact = _load_model_from_source(
            config=config,
            hf_revision_override=hf_revision_override,
            device=device,
        )

    validate_runtime_compatibility(runtime_config=config, artifact=artifact)

    landmark_extractor = _build_landmark_extractor(config_root)
    hand_selector = HandSelector(
        hand_preference=config.app.hand_preference,
        min_handedness_conf=0.7,
        min_track_conf=0.5,
    )
    feature_transformer: IFeatureTransformer = LandmarkFeatureTransformer(
        enable_orientation_normalization=config.feature_pipeline.orientation_normalization,
    )
    sequence_buffer: ISequenceBuffer = SlidingWindowSequenceBuffer(
        window_length=config.inference.window_length,
        window_step=config.inference.window_step,
        min_valid_ratio=config.inference.min_valid_ratio,
    )

    stabilizer = PredictionStabilizer(
        default_threshold=float(thresholds.get("acceptance_default", 0.85)),
        class_thresholds=thresholds.get("class_thresholds", {}),
        vote_window=3,
        class_vote_windows=thresholds.get("class_vote_windows") or None,
        ema_alpha=0.4,
        unknown_label="unknown",
    )
    activation_controller = GestureActivationController(
        command_map=command_map,
        command_cooldown_ms=config.inference.command_cooldown_ms,
        edge_trigger_enabled=config.inference.edge_trigger_enabled,
        continuous_labels=continuous_labels,
        unknown_label="unknown",
    )
    dispatcher = _build_dispatcher(
        allowed_commands=frozenset(command_map.values()),
    )
    overlay_renderer = OverlayRenderer(
        mirror_display=config.app.mirror_display,
        command_map=command_map,
        # Produce RGB output when the preview window is active so Qt widgets
        # (QImage.Format_RGB888) receive frames in the correct channel order
        # without an additional per-frame copy.
        output_rgb=show_window,
        minimal_preview=minimal_preview,
    )
    benchmark = LatencyBenchmark()

    return AppContainer(
        config=config,
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        hand_selector=hand_selector,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
        model_artifact=artifact,
        stabilizer=stabilizer,
        activation_controller=activation_controller,
        dispatcher=dispatcher,
        overlay_renderer=overlay_renderer,
        benchmark=benchmark,
        show_window=show_window,
        label_remap=label_remap,
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


def _select_device() -> torch.device:
    """Select the best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _build_landmark_extractor(config_root: Path) -> IHandLandmarkExtractor:
    """Build hand landmark extractor with Tasks GPU path when available.

    Falls back to the classic MediaPipe solutions extractor if the task model
    file is unavailable or Tasks initialization fails.
    """
    use_tasks = os.getenv("MAESTRO_USE_MEDIAPIPE_TASKS", "0") == "1"
    task_model = config_root.parent / "models" / "mediapipe" / "hand_landmarker.task"
    if use_tasks and task_model.exists():
        try:
            return MediaPipeTasksHandsExtractor(
                model_path=task_model,
                use_gpu_delegate=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except Exception:
            # Safe fallback for environments where the GPU delegate cannot be used.
            pass

    return MediaPipeHandsExtractor(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def _load_model_from_source(
    *,
    config: Any,
    hf_revision_override: str | None,
    device: torch.device,
) -> InferenceArtifact:
    """Load the enhanced two-stream attention artifact from Hugging Face Hub.

    When *hf_revision_override* is provided it overrides ``model_source.revision``
    (supports the ``--hf-revision`` CLI flag).
    """
    src = config.model_source
    from maestro.infrastructure.model.hf_loader import (
        load_artifact_from_hub,
    )

    return load_artifact_from_hub(
        repo_id=src.repo_id,
        filename=src.filename,
        revision=hf_revision_override or src.revision,
        cache_dir=src.cache_dir,
        device=device,
    )


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


def _load_command_map(path: Path) -> tuple[dict[str, str], frozenset[str], dict[str, str]]:
    """Load gesture→command_id mapping, gesture mode classifications, and label
    remapping from YAML.

    Returns:
        A tuple of ``(command_map, continuous_labels, label_remap)`` where
        *command_map* maps gesture label → command_id for discrete gestures,
        *continuous_labels* is the frozenset of gesture labels classified as
        ``"continuous"`` in the optional ``gesture_modes`` block, and
        *label_remap* is an optional dict that maps model output labels to
        their display/dispatch equivalents (e.g. correcting for mirror-view
        swipe direction without retraining).

    Raises:
        FileNotFoundError: If the command map file does not exist.
        ValueError: If the file content is not a valid YAML mapping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Command map file not found: {path}. "
            "Ensure configs/gestures/command_map_presentation.yaml exists."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Command map file must contain a YAML mapping: {path}")
    mapping = payload.get("gesture_commands", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"Command map file must contain a 'gesture_commands' mapping: {path}")
    modes = payload.get("gesture_modes", {})
    if not isinstance(modes, dict):
        modes = {}
    continuous_labels = frozenset(k for k, v in modes.items() if str(v) == "continuous")
    command_map = {str(k): str(v) for k, v in mapping.items() if k and v}
    raw_remap = payload.get("label_remap", {})
    label_remap = (
        {str(k): str(v) for k, v in raw_remap.items() if k and v}
        if isinstance(raw_remap, dict)
        else {}
    )
    return command_map, continuous_labels, label_remap


def _build_dispatcher(
    allowed_commands: frozenset[str],
) -> CommandDispatcher:
    """Wrap a PresentationAdapter in a CommandDispatcher."""
    adapter = PresentationAdapter(
        keyboard=KeyboardAdapter(),
    )
    return CommandDispatcher(
        adapter=adapter,
        allowed_commands=allowed_commands,
    )
