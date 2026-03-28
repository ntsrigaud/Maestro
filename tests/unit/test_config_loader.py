from __future__ import annotations

from pathlib import Path

from gesture_control.utils.config_loader import load_runtime_config


def test_load_runtime_config(config_dir: Path) -> None:
    config = load_runtime_config(config_dir)
    assert config.app.app_name == "presentation-meeting-control"
    assert config.inference.window_length == 24
    assert config.inference.window_step == 3
    assert config.feature_pipeline.orientation_normalization is True
