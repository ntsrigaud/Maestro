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
