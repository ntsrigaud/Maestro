# Work Log — Desktop Interaction: Phase 1 Configuration Foundation

**Date:** 2026-04-19  
**Branch:** `build/configuration-foundation`  
**Author:** @ntsrigaud

---

## Context

Phase 1 configuration work from the desktop interaction roadmap. The branch establishes
the desktop gesture-command vocabulary, extends `AppConfig` with the fields required for
config-driven command map selection and horizontal swipe routing, fixes the hardcoded
command map path in `bootstrap.py`, and covers all new configuration paths with a unit
test suite. `DesktopConfig` schema fields (`scroll_clicks`, `zoom_clicks`, `linux_de`) are
deferred to Phase 3.

---

## Technical Changes

### `configs/gestures/command_map_desktop.yaml` (new)

Application-agnostic gesture-to-command map for the desktop profile. All 8 IPN gesture
classes mapped to logical command IDs:

| Gesture                      | Command                              |
| ---------------------------- | ------------------------------------ |
| `swipe_up` / `swipe_down`    | `scroll_forward` / `scroll_back`     |
| `swipe_right` / `swipe_left` | `navigate_forward` / `navigate_back` |
| `zoom_in` / `zoom_out`       | `zoom_in_view` / `zoom_out_view`     |
| `open_palm_hold`             | `mission_control`                    |
| `unknown`                    | `no_action`                          |

Existing `command_map_ipn.yaml` left untouched.

### `configs/inference_desktop.yaml` (new)

Reference timing profile documenting desktop-tuned inference values
(`command_cooldown_ms: 800`, `idle_timeout_ms: 60000`, `gaze.grace_ms: 3000`).
Annotated as a reference artefact — not loaded at runtime; the active desktop config
is `configs/desktop/inference.yaml`.

### `configs/app.yaml`

Two new fields added with backwards-compatible defaults:

```yaml
command_map: command_map_ipn.yaml # switch to command_map_desktop.yaml for desktop
navigation_mode: workspace # workspace | tabs
```

### `src/gesture_control/utils/schema_validation.py`

**`AppConfig` extensions:**

- `command_map: str` — selects the active gesture map filename from `configs/gestures/`;
  defaults to `"command_map_ipn.yaml"` for backwards compatibility.
- `navigation_mode: str` — constrained by `pattern="^(workspace|tabs)$"`; controls
  horizontal swipe destination in `DesktopAdapter`.
- `desktop: "DesktopConfig"` — forward reference to the `DesktopConfig` model defined in
  Phase 3; field is wired here so the schema slot exists before Phase 3 lands.

### `src/gesture_control/app/bootstrap.py`

Command map path de-hardcoded:

```python
# Before
command_map = _load_command_map(config_root / "gestures" / "command_map_ipn.yaml")

# After
command_map = _load_command_map(config_root / "gestures" / config.app.command_map)
```

Without this fix, `command_map_desktop.yaml` is never loaded regardless of `app.yaml`.

---

## Testing

**`tests/unit/test_phase1_config.py`** (182 lines, new):

| Test class                          | Coverage                                                                                                                                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TestAppConfigNewFields`            | Default values; `command_map` override; `navigation_mode` accepts `workspace`/`tabs`, rejects invalid values; unknown fields rejected; backwards compatibility with existing `configs/app.yaml` |
| `TestCommandMapDesktop`             | `command_map_desktop.yaml` loads without schema errors; contains all 8 gesture keys and expected command IDs; `version: v2` present                                                             |
| `TestInferenceDesktopYaml`          | `inference_desktop.yaml` validates against `InferenceConfig`; `command_cooldown_ms`, `gaze.grace_ms`, `window_length` correct                                                                   |
| `TestBootstrapCommandMapResolution` | Bootstrap resolves `config.app.command_map` to the desktop vocabulary when `app.yaml` specifies `command_map_desktop.yaml`; `_load_command_map` returns desktop command IDs                     |

---

## Outcome

Phase 1 of the desktop interaction configuration layer is complete. `AppConfig` accepts
`command_map` and `navigation_mode` with backwards-compatible defaults; the bootstrap
resolves the command map filename from config rather than a hardcoded string; the full IPN
presentation profile loads unchanged. Phase 2 (`DesktopAdapter` implementation) and
Phase 3 (`DesktopConfig` schema and `configs/desktop/` profile directory) can proceed
independently from this foundation.
