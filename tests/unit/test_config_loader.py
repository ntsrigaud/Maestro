from __future__ import annotations

from pathlib import Path

from maestro.utils.config_loader import load_runtime_config


def test_load_runtime_config(config_dir: Path) -> None:
    config = load_runtime_config(config_dir)
    assert config.app.app_name == "presentation-controller"
    assert config.inference.window_length == 16
    assert config.inference.window_step == 3
    assert config.inference.pointer.gain_x == 3.0
    assert config.inference.pointer.gain_y == 2.8
    assert len(config.inference.movement_range_presets) == 3
    assert config.inference.movement_range_presets[0].key == "low"
    assert config.inference.movement_range_presets[1].key == "medium"
    assert config.inference.movement_range_presets[2].key == "high"
    assert config.inference.movement_range_presets[2].gain_x == 3.0
    assert config.inference.default_movement_range == "high"
    assert config.feature_pipeline.orientation_normalization is False
