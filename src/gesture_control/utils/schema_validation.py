from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    camera_index: int = Field(ge=0)
    frame_width: int = Field(ge=320)
    frame_height: int = Field(ge=240)
    target_fps: int = Field(ge=1, le=120)
    active_app: str
    # When True the display window is flipped horizontally so the preview acts
    # like a selfie/mirror camera.  MediaPipe landmark extraction and the entire
    # ML pipeline always operate on the original non-mirrored frame.
    mirror_display: bool = True


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_length: int = Field(ge=4, le=256)
    window_step: int = Field(ge=1, le=64)
    min_valid_ratio: float = Field(ge=0.0, le=1.0)
    activation_hold_ms: int = Field(ge=100, le=5000)
    # How long the system stays ARMED with no recognised gesture before
    # dropping back to IDLE.  Default 300 000 ms (5 min) suits presentation
    # use where users pause between interactions for extended periods.
    idle_timeout_ms: int = Field(ge=500, le=600_000)
    command_cooldown_ms: int = Field(ge=100, le=5000)


class FeaturePipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wrist_relative_normalization: bool = True
    scale_normalization: bool = True
    orientation_normalization: bool = False
    velocity_features: bool = True


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    inference: InferenceConfig
    feature_pipeline: FeaturePipelineConfig = FeaturePipelineConfig()
