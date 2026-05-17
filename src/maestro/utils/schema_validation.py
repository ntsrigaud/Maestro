from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PointerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gain_x: float = Field(default=2.4, ge=1.0, le=5.0)
    gain_y: float = Field(default=2.2, ge=1.0, le=5.0)


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
    # Shared pointer sensitivity used by both the laser pointer and the
    # annotation pen continuous trackers.
    pointer: PointerConfig = PointerConfig()


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
