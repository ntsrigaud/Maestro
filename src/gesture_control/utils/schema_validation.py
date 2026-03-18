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


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_length: int = Field(ge=4, le=256)
    window_step: int = Field(ge=1, le=64)
    min_valid_ratio: float = Field(ge=0.0, le=1.0)
    activation_hold_ms: int = Field(ge=100, le=5000)
    idle_timeout_ms: int = Field(ge=500, le=15000)
    command_cooldown_ms: int = Field(ge=100, le=5000)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    inference: InferenceConfig
