from __future__ import annotations

from pathlib import Path

from maestro.utils.config_loader import load_runtime_config


def test_load_runtime_config(config_dir: Path) -> None:
    config = load_runtime_config(config_dir)
    assert config.app.app_name == "presentation-controller"
    assert config.inference.window_length == 16
    assert config.inference.window_step == 3
    assert config.inference.pointer.gain_x == 2.4
    assert config.inference.pointer.gain_y == 2.2
    assert config.feature_pipeline.orientation_normalization is False
