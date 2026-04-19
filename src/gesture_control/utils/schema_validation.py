from __future__ import annotations

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
    # Selects the active gesture-to-command map filename (without path prefix).
    # The file must exist in configs/gestures/.  Defaults to the IPN
    # presentation map for backwards compatibility.
    command_map: str = "command_map_ipn.yaml"
    # Controls horizontal swipe destination for the desktop adapter.
    # "workspace" → OS virtual desktop shortcuts (Ctrl+→ / Ctrl+Win+→ etc.)
    # "tabs"      → Ctrl+Tab / Ctrl+Shift+Tab (browser/editor tabs)
    navigation_mode: str = Field(default="workspace", pattern="^(workspace|tabs)$")
    # Desktop-specific fine-tuning; ignored when active_app != "desktop".
    desktop: DesktopConfig = Field(default_factory=lambda: DesktopConfig())


class DesktopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Number of pyautogui scroll clicks per scroll gesture event.
    # Tune for comfortable document scrolling at your display DPI / resolution.
    scroll_clicks: int = Field(default=5, ge=1, le=20)
    # Number of modifier+scroll steps per zoom gesture event.
    # Smaller values give finer zoom control; larger give faster zoom.
    zoom_clicks: int = Field(default=3, ge=1, le=10)
    # Linux desktop environment name, used to select the correct Activities /
    # Overview shortcut.  Case-insensitive.  Supported: gnome (default), kde.
    # Any other value falls back to the GNOME shortcut (Super+A).
    linux_de: str = Field(default="gnome", pattern="^[a-z][a-z0-9_-]{0,31}$")


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

    # "local" loads from the local registry JSON; "huggingface" downloads from HF Hub.
    backend: str = Field(default="local", pattern="^(local|huggingface)$")
    # Hugging Face Hub settings (required when backend == "huggingface").
    repo_id: str = "ntsrigaud/gesture-control-lstm"
    revision: str = "main"
    filename: str = "lstm_v1_inference.pt"
    cache_dir: str | None = None
    # Local registry settings (used when backend == "local").
    registry_path: str = "models/registry.json"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig
    inference: InferenceConfig
    feature_pipeline: FeaturePipelineConfig = Field(default_factory=FeaturePipelineConfig)
    model_source: ModelSourceConfig = Field(default_factory=ModelSourceConfig)
