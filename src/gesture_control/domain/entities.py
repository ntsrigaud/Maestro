from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FramePacket:
    frame: Any
    timestamp_ms: int
    frame_id: int


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
    vector: list[float]
    timestamp_ms: int
    valid: bool


@dataclass(slots=True)
class SequenceWindow:
    features: list[list[float]]
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
    gaze_permitted: bool = True
    info: dict[str, Any] = field(default_factory=dict)
