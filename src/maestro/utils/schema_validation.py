from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GazeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Master enable.  When False, a NullGazeDetector (always returns True) is
    # substituted, restoring the pre-gaze behaviour without code changes.
    enabled: bool = True
    # Head-pose acceptance cone.  A face is considered camera-directed when
    # both the absolute yaw and absolute pitch remain within these limits.
    yaw_threshold_deg: float = Field(default=30.0, gt=0, le=90)
    pitch_threshold_deg: float = Field(default=25.0, gt=0, le=90)
    # Duration (ms) the head pose must remain within the acceptance cone
    # continuously before the detector transitions to GAZE_ACTIVE.
    arm_ms: int = Field(default=500, ge=100, le=5000)
    # Grace period (ms) after the head pose leaves the acceptance cone before
    # the system forces the activation controller back to IDLE.  Prevents
    # false disarming during natural head movements.
    grace_ms: int = Field(default=5000, ge=1000, le=60_000)
    # Face Mesh is called every N frames; the cached result is returned for
    # intermediate frames.  At 30 FPS and N=3, Face Mesh runs at ~10 Hz.
    sample_interval_frames: int = Field(default=3, ge=1, le=30)
    # Minimum BlazeFace detection confidence score [0, 1] required before a
    # keypoint-based head-pose check is attempted.  Lower values accept more
    # detections at range but increase false-positive cone membership.
    min_detection_score: float = Field(default=0.5, gt=0.0, le=1.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    camera_index: int = Field(ge=0)
    frame_width: int = Field(ge=320)
    frame_height: int = Field(ge=240)
    target_fps: int = Field(ge=1, le=120)
    # When True the display window is flipped horizontally so the preview acts
    # like a selfie/mirror camera.  MediaPipe landmark extraction and the entire
    # ML pipeline always operate on the original non-mirrored frame.
    mirror_display: bool = True
    # Selects the active gesture-to-command map filename (without path prefix).
    # The file must exist in configs/gestures/.
    command_map: str = "command_map_presentation.yaml"
    # Restricts gesture recognition to a specific physical hand from the user's
    # perspective.  Accepted values:
    #   "right" — only recognise the user's right hand.
    #   "left"  — only recognise the user's left hand.
    #   "both"  — accept whichever hand is detected (default behaviour).
    # Internally the extractor inverts the label before comparing with MediaPipe,
    # because MediaPipe assumes a mirrored/selfie frame: in a standard non-mirrored
    # webcam feed the user's right hand is reported as "Left" and vice versa.
    hand_preference: Literal["left", "right", "both"] = "both"


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_length: int = Field(ge=4, le=256)
    window_step: int = Field(ge=1, le=64)
    min_valid_ratio: float = Field(ge=0.0, le=1.0)
    # How long the system stays ARMED with no recognised gesture before
    # dropping back to IDLE.  Default 300 000 ms (5 min) suits presentation
    # use where users pause between interactions for extended periods.
    idle_timeout_ms: int = Field(ge=500, le=600_000)
    command_cooldown_ms: int = Field(ge=100, le=5000)
    # When True (default), a command gesture is only dispatched on its rising
    # edge — the first frame where the label *changes* from the previous label.
    # This prevents repeated commands from a single slow gesture that produces
    # many consecutive accepted predictions across overlapping inference windows.
    edge_trigger_enabled: bool = True
    # Gaze-aware activation.  See GazeConfig for field documentation.
    gaze: GazeConfig = GazeConfig()


class FeaturePipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wrist_relative_normalization: bool = True
    scale_normalization: bool = True
    orientation_normalization: bool = False
    velocity_features: bool = True


class ModelSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="huggingface", pattern="^huggingface$")
    # Hugging Face Hub settings.
    repo_id: str = "ntsrigaud/maestro-lstm-hybrid"
    revision: str = "main"
    filename: str = "two_stream_attn_v1_2layer_ld_cong_finetune_20260515T134706Z_inference.pt"
    cache_dir: str | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    inference: InferenceConfig
    feature_pipeline: FeaturePipelineConfig = Field(default_factory=FeaturePipelineConfig)
    model_source: ModelSourceConfig = Field(default_factory=ModelSourceConfig)
