from __future__ import annotations

import yaml

from maestro.app.ui.config_panel import ConfigPanel


class TestConfigPanelValidation:
    def test_panel_loads_without_error(self, qtbot, tmp_path) -> None:
        """ConfigPanel constructs even when config files are absent."""
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)

    def test_apply_button_disabled_while_runtime_active(self, qtbot, tmp_path) -> None:
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        panel.set_runtime_active(True)
        assert not panel._apply_btn.isEnabled()

    def test_apply_button_enabled_when_valid_and_stopped(self, qtbot, tmp_path) -> None:
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        panel.set_runtime_active(False)
        # Trigger validation refresh
        panel._on_any_change()
        # Apply button state depends on validation; if widgets present but base
        # config missing, valid=False and button is disabled. Both sides equal.
        if panel._widgets:
            assert panel._apply_btn.isEnabled() == panel._valid

    def test_config_changed_signal_emitted_on_change(self, qtbot, tmp_path) -> None:
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.config_changed, timeout=500):
            panel._on_any_change()

    def test_current_config_returns_none_when_invalid(self, qtbot, tmp_path) -> None:
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        panel._valid = False
        assert panel.current_config() is None

    def test_safe_write_is_atomic(self, tmp_path) -> None:
        from maestro.app.ui.config_panel import _safe_write

        target = tmp_path / "test.yaml"
        _safe_write(target, {"key": "value"})
        result = yaml.safe_load(target.read_text())
        assert result == {"key": "value"}
        # Temp file must not exist after successful write
        assert not (tmp_path / "test.yaml.tmp").exists()


class TestGazeToggle:
    """Verify that the gaze_enabled checkbox round-trips through config save/load."""

    def _write_valid_configs(self, tmp_path, gaze_enabled: bool) -> None:
        app_cfg = {
            "app_name": "test-app",
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "target_fps": 30,
            "mirror_display": True,
            "command_map": "command_map_presentation.yaml",
            "hand_preference": "right",
        }
        infer_cfg = {
            "window_length": 32,
            "window_step": 3,
            "min_valid_ratio": 0.8,
            "idle_timeout_ms": 100000,
            "command_cooldown_ms": 1000,
            "edge_trigger_enabled": True,
            "gaze": {
                "enabled": gaze_enabled,
                "yaw_threshold_deg": 30.0,
                "pitch_threshold_deg": 25.0,
                "arm_ms": 500,
                "grace_ms": 5000,
                "sample_interval_frames": 3,
                "min_detection_score": 0.5,
            },
        }
        (tmp_path / "app.yaml").write_text(yaml.dump(app_cfg))
        (tmp_path / "inference.yaml").write_text(yaml.dump(infer_cfg))

    def test_gaze_enabled_widget_present(self, qtbot, tmp_path) -> None:
        """The gaze_enabled widget is always created by _build_gaze_group."""
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        assert "gaze_enabled" in panel._widgets

    def test_gaze_enabled_populated_from_yaml_false(self, qtbot, tmp_path) -> None:
        self._write_valid_configs(tmp_path, gaze_enabled=False)
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        from PySide6.QtWidgets import QCheckBox

        cb = panel._widgets["gaze_enabled"]
        assert isinstance(cb, QCheckBox)
        assert not cb.isChecked()

    def test_gaze_enabled_populated_from_yaml_true(self, qtbot, tmp_path) -> None:
        self._write_valid_configs(tmp_path, gaze_enabled=True)
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        from PySide6.QtWidgets import QCheckBox

        cb = panel._widgets["gaze_enabled"]
        assert isinstance(cb, QCheckBox)
        assert cb.isChecked()

    def test_toggling_gaze_writes_correct_value(self, qtbot, tmp_path) -> None:
        """Flipping the checkbox and applying writes the new value to YAML."""
        self._write_valid_configs(tmp_path, gaze_enabled=False)
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        from PySide6.QtWidgets import QCheckBox

        cb = panel._widgets["gaze_enabled"]
        assert isinstance(cb, QCheckBox)
        cb.setChecked(True)
        panel._on_apply()
        written = yaml.safe_load((tmp_path / "inference.yaml").read_text())
        assert written["gaze"]["enabled"] is True

    def test_gaze_yaml_fields_preserved_after_toggle(self, qtbot, tmp_path) -> None:
        """Advanced gaze fields (thresholds, timing) survive a toggle+apply cycle."""
        self._write_valid_configs(tmp_path, gaze_enabled=False)
        panel = ConfigPanel(config_path=str(tmp_path))
        qtbot.addWidget(panel)
        from PySide6.QtWidgets import QCheckBox

        cb = panel._widgets["gaze_enabled"]
        assert isinstance(cb, QCheckBox)
        cb.setChecked(True)
        panel._on_apply()
        written = yaml.safe_load((tmp_path / "inference.yaml").read_text())
        assert written["gaze"]["arm_ms"] == 500
        assert written["gaze"]["grace_ms"] == 5000
        assert written["gaze"]["yaw_threshold_deg"] == 30.0
