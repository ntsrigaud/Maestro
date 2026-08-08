from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FramePacket:
    frame: Any  # BGR ndarray (OpenCV native)
    timestamp_ms: int
    frame_id: int
    rgb_frame: np.ndarray | None = None  # pre-converted C-contiguous RGB; None if not available


@dataclass(slots=True)
class HandLandmarks:
    """Landmarks and quality metadata for a single detected hand."""

    label: str  # MediaPipe label: "Left" or "Right" (selfie-view convention)
    landmarks: list[list[float]]  # 21 × [x, y, z] in normalised image coordinates
    detect_conf: float  # MediaPipe handedness classification score [0, 1]
    track_conf: float  # fraction of finite landmark coordinates [0, 1]


@dataclass(slots=True)
class MultiHandPacket:
    """Raw output from the hand landmark extractor: data for all detected hands.

    ``hands`` contains 0–2 entries (one per MediaPipe detection).  The order
    mirrors MediaPipe's internal detection ordering (highest confidence first).
    Hand-preference selection is deliberately deferred to the pipeline
    (``HandSelector``) so the extractor remains stateless and preference-agnostic.
    """

    hands: list[HandLandmarks]
    timestamp_ms: int


@dataclass(slots=True)
class LandmarkPacket:
    landmarks: list[list[float]]
    handedness: str | None
    detect_conf: float
    track_conf: float
    timestamp_ms: int
    valid: bool


@dataclass(slots=True)
class FeatureFrame:
    vector: np.ndarray
    timestamp_ms: int
    valid: bool


@dataclass(slots=True)
class SequenceWindow:
    features: np.ndarray
    start_timestamp_ms: int
    end_timestamp_ms: int
    valid_ratio: float


@dataclass(slots=True)
class Prediction:
    class_probs: dict[str, float]
    top_class: str
    confidence: float


@dataclass(slots=True)
class StabilizedPrediction:
    label: str
    confidence: float
    accepted: bool
    reason: str | None = None


@dataclass(slots=True)
class ExecutableCommand:
    command_id: str
    source_gesture: str
    issued_at_ms: int


@dataclass(slots=True)
class DispatchResult:
    success: bool
    command_id: str
    target_app: str
    detail: str | None = None


@dataclass(slots=True)
class RuntimeState:
    mode: str
    last_label: str = "unknown"
    last_confidence: float = 0.0
    active_mode: str = "unknown"
    app_focused: bool = True
    target_app_name: str = "PowerPoint"
    info: dict[str, Any] = field(default_factory=dict)
    # Rendered BGR frame for display.  Populated by Runtime when show_window is
    # True; None otherwise.  Consumers (GUI PreviewWidget or CLI on_state
    # callback) display this in their own thread-safe context.
    preview_frame: Any = field(default=None)
