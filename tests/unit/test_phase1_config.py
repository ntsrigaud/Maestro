"""Phase 1 & 3 tests — Configuration Foundation.

Phase 1 covers:
- AppConfig accepts command_map and navigation_mode fields with correct defaults.
- AppConfig rejects invalid navigation_mode values.
- AppConfig without new fields in YAML loads successfully (backwards compatibility).
- command_map_desktop.yaml loads and contains the expected command vocabulary.
- inference_desktop.yaml validates against InferenceConfig.
- bootstrap command map resolution uses config.app.command_map.

"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
        assert cfg.command_map == "command_map_ipn.yaml"
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
        with pytest.raises(Exception):
            AppConfig.model_validate(self._minimal(navigation_mode="context_aware"))

    def test_unknown_field_still_rejected(self):
        with pytest.raises(Exception):
            AppConfig.model_validate(self._minimal(dispatch_mode="static"))

    def test_backwards_compat_existing_configs(self, config_dir: Path):
        """Existing configs/app.yaml must load without errors."""
        config = load_runtime_config(config_dir)
        assert config.app.command_map == "command_map_ipn.yaml"
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
        assert cmds["swipe_up"] == "scroll_forward"
        assert cmds["swipe_down"] == "scroll_back"

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

