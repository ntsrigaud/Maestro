"""Phase 1 & 3 tests — Configuration Foundation.

Phase 1 covers:
- AppConfig accepts command_map and navigation_mode fields with correct defaults.
- AppConfig rejects invalid navigation_mode values.
- AppConfig without new fields in YAML loads successfully (backwards compatibility).
- command_map_desktop.yaml loads and contains the expected command vocabulary.
- inference_desktop.yaml validates against InferenceConfig.
- bootstrap command map resolution uses config.app.command_map.

Phase 3 covers:
- DesktopConfig validates scroll_clicks, zoom_clicks, and linux_de.
- AppConfig.desktop defaults are correct.
- configs/desktop/ directory loads via load_runtime_config.
- configs/desktop/app.yaml sets active_app=desktop and command_map_desktop.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gesture_control.utils.config_loader import load_runtime_config
from gesture_control.utils.schema_validation import AppConfig, DesktopConfig, InferenceConfig

# ---------------------------------------------------------------------------
# AppConfig schema extension
# ---------------------------------------------------------------------------


class TestAppConfigNewFields:
    def _minimal(self, **overrides) -> dict:
        base = {
            "app_name": "test-app",
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "target_fps": 30,
            "active_app": "desktop",
        }
        base.update(overrides)
        return base

    def test_defaults_applied_when_fields_absent(self):
        cfg = AppConfig.model_validate(self._minimal())
        assert cfg.command_map == "command_map_desktop.yaml"
        assert cfg.navigation_mode == "workspace"

    def test_command_map_override(self):
        cfg = AppConfig.model_validate(
            self._minimal(command_map="command_map_desktop.yaml")
        )
        assert cfg.command_map == "command_map_desktop.yaml"

    def test_navigation_mode_workspace_accepted(self):
        cfg = AppConfig.model_validate(self._minimal(navigation_mode="workspace"))
        assert cfg.navigation_mode == "workspace"

    def test_navigation_mode_tabs_accepted(self):
        cfg = AppConfig.model_validate(self._minimal(navigation_mode="tabs"))
        assert cfg.navigation_mode == "tabs"

    def test_navigation_mode_invalid_rejected(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate(self._minimal(navigation_mode="context_aware"))

    def test_unknown_field_still_rejected(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate(self._minimal(dispatch_mode="static"))

    def test_backwards_compat_existing_configs(self, config_dir: Path):
        """Existing configs/app.yaml must load without errors."""
        config = load_runtime_config(config_dir)
        assert config.app.command_map == "command_map_desktop.yaml"
        assert config.app.navigation_mode == "workspace"


# ---------------------------------------------------------------------------
# command_map_desktop.yaml
# ---------------------------------------------------------------------------


class TestCommandMapDesktop:
    def _load_desktop_map(self, config_dir: Path) -> dict:
        path = config_dir / "gestures" / "command_map_desktop.yaml"
        assert path.exists(), f"command_map_desktop.yaml not found at {path}"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_file_exists_and_parses(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        assert isinstance(data, dict)

    def test_version_field_present(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        assert data.get("version") == "v2"

    def test_all_eight_gestures_mapped(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        gestures = data["gesture_commands"]
        expected_gestures = {
            "swipe_up", "swipe_down", "swipe_right", "swipe_left",
            "zoom_in", "zoom_out", "open_palm_hold", "unknown",
        }
        assert set(gestures.keys()) == expected_gestures

    def test_mission_control_mapped(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        assert data["gesture_commands"]["open_palm_hold"] == "mission_control"

    def test_scroll_commands_present(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        cmds = data["gesture_commands"]
        assert cmds["swipe_up"] == "scroll_back"
        assert cmds["swipe_down"] == "scroll_forward"

    def test_zoom_commands_present(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        cmds = data["gesture_commands"]
        assert cmds["zoom_in"] == "zoom_in_view"
        assert cmds["zoom_out"] == "zoom_out_view"

    def test_navigation_commands_present(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        cmds = data["gesture_commands"]
        assert cmds["swipe_right"] == "navigate_forward"
        assert cmds["swipe_left"] == "navigate_back"

    def test_unknown_maps_to_no_action(self, config_dir: Path):
        data = self._load_desktop_map(config_dir)
        assert data["gesture_commands"]["unknown"] == "no_action"


# ---------------------------------------------------------------------------
# inference_desktop.yaml (reference artefact)
# ---------------------------------------------------------------------------


class TestInferenceDesktopYaml:
    def test_validates_against_inference_config(self, config_dir: Path):
        path = config_dir.parent / "configs" / "inference_desktop.yaml"
        # Also try relative to the config_dir itself (test fixture may vary)
        if not path.exists():
            path = config_dir / "inference_desktop.yaml"
        if not path.exists():
            # Resolve from repo root
            path = config_dir.parent / "inference_desktop.yaml"
        assert path.exists(), f"inference_desktop.yaml not found; tried {path}"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = InferenceConfig.model_validate(data)
        assert cfg.window_length == 24
        assert cfg.command_cooldown_ms == 800
        assert cfg.gaze.grace_ms == 3000


# ---------------------------------------------------------------------------
# Bootstrap command map resolution
# ---------------------------------------------------------------------------


class TestBootstrapCommandMapResolution:
    def test_desktop_map_loaded_when_config_specifies_it(self, config_dir: Path, tmp_path: Path):
        """When app.yaml names command_map_desktop.yaml the bootstrap must load
        the desktop vocabulary (not the hardcoded IPN map)."""
        import shutil

        # Create a minimal config directory with app.yaml pointing to desktop map
        cfg_dir = tmp_path / "configs"
        shutil.copytree(config_dir, cfg_dir)
        app_yaml = cfg_dir / "app.yaml"
        app_data = yaml.safe_load(app_yaml.read_text(encoding="utf-8"))
        app_data["command_map"] = "command_map_desktop.yaml"
        app_data["active_app"] = "desktop"
        app_yaml.write_text(yaml.dump(app_data), encoding="utf-8")

        config = load_runtime_config(cfg_dir)
        assert config.app.command_map == "command_map_desktop.yaml"

        # Verify _load_command_map would resolve to the desktop map
        from gesture_control.app.bootstrap import _load_command_map  # type: ignore[attr-defined]
        result = _load_command_map(cfg_dir / "gestures" / config.app.command_map)
        assert "mission_control" in result.values()
        assert "no_action" in result.values()
        assert "scroll_forward" in result.values()


# ---------------------------------------------------------------------------
# Phase 3: DesktopConfig schema
# ---------------------------------------------------------------------------


class TestDesktopConfig:
    def test_defaults(self):
        cfg = DesktopConfig()
        assert cfg.scroll_clicks == 5
        assert cfg.zoom_clicks == 3
        assert cfg.linux_de == "gnome"

    def test_scroll_clicks_bounds(self):
        assert DesktopConfig(scroll_clicks=1).scroll_clicks == 1
        assert DesktopConfig(scroll_clicks=20).scroll_clicks == 20
        with pytest.raises(ValidationError):
            DesktopConfig(scroll_clicks=0)
        with pytest.raises(ValidationError):
            DesktopConfig(scroll_clicks=21)

    def test_zoom_clicks_bounds(self):
        assert DesktopConfig(zoom_clicks=1).zoom_clicks == 1
        assert DesktopConfig(zoom_clicks=10).zoom_clicks == 10
        with pytest.raises(ValidationError):
            DesktopConfig(zoom_clicks=0)
        with pytest.raises(ValidationError):
            DesktopConfig(zoom_clicks=11)

    def test_linux_de_pattern_accepted(self):
        for de in ("gnome", "kde", "xfce", "cinnamon", "lxde"):
            assert DesktopConfig(linux_de=de).linux_de == de

    def test_linux_de_invalid_pattern_rejected(self):
        with pytest.raises(ValidationError):
            DesktopConfig(linux_de="GNOME!")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            DesktopConfig.model_validate({"scroll_clicks": 5, "unknown_field": True})


class TestAppConfigDesktopField:
    def _minimal(self, **overrides) -> dict:
        base = {
            "app_name": "test",
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "target_fps": 30,
            "active_app": "desktop",
        }
        base.update(overrides)
        return base

    def test_desktop_defaults_nested(self):
        cfg = AppConfig.model_validate(self._minimal())
        assert cfg.desktop.scroll_clicks == 5
        assert cfg.desktop.zoom_clicks == 3
        assert cfg.desktop.linux_de == "gnome"

    def test_desktop_overrides_respected(self):
        cfg = AppConfig.model_validate(
            self._minimal(desktop={"scroll_clicks": 10, "zoom_clicks": 5, "linux_de": "kde"})
        )
        assert cfg.desktop.scroll_clicks == 10
        assert cfg.desktop.zoom_clicks == 5
        assert cfg.desktop.linux_de == "kde"


# ---------------------------------------------------------------------------
# Phase 3: configs/desktop/ directory
# ---------------------------------------------------------------------------


class TestDesktopConfigDirectory:
    def _desktop_config_dir(self, config_dir: Path) -> Path:
        return config_dir.parent / "configs" / "desktop"

    def test_desktop_dir_exists(self, config_dir: Path):
        desktop_dir = self._desktop_config_dir(config_dir)
        assert desktop_dir.is_dir(), f"configs/desktop/ not found at {desktop_dir}"

    def test_desktop_app_yaml_loads_successfully(self, config_dir: Path):
        desktop_dir = self._desktop_config_dir(config_dir)
        config = load_runtime_config(desktop_dir)
        assert config.app.active_app == "desktop"
        assert config.app.command_map == "command_map_desktop.yaml"
        assert config.app.navigation_mode == "workspace"

    def test_desktop_inference_yaml_loads_successfully(self, config_dir: Path):
        desktop_dir = self._desktop_config_dir(config_dir)
        config = load_runtime_config(desktop_dir)
        assert config.inference.window_length == 24
        assert config.inference.command_cooldown_ms == 800
        assert config.inference.gaze.grace_ms == 3000

    def test_desktop_config_desktop_field_loaded(self, config_dir: Path):
        desktop_dir = self._desktop_config_dir(config_dir)
        config = load_runtime_config(desktop_dir)
        assert config.app.desktop.scroll_clicks == 5
        assert config.app.desktop.zoom_clicks == 3
        assert config.app.desktop.linux_de == "gnome"
