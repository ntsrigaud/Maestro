# Desktop Interaction Expansion — Implementation Roadmap

**Date:** 2026-04-19  
**Branch target:** `feature/desktop-interaction`  
**Analysis source:** [`desktop-interaction-analysis.md`](desktop-interaction-analysis.md)  
**Scope:** Extend the gesture control system to support close-range desktop interaction (document scrolling, tab navigation, workspace switching) without modifying the core pipeline, model, or training infrastructure.

---

## Guiding Constraints

1. **No pipeline changes.** Feature engineering, LSTM, stabiliser, and gaze activation are untouched.
2. **No retraining.** The production model (Macro-F1 = 0.96) is adequate for all desktop scenarios at desk distance.
3. **Single unified adapter.** One `DesktopAdapter` replaces the original three-adapter plan. Scroll is application-agnostic (`pyautogui.scroll()`); navigation and overview use per-platform shortcut tables with no app-process detection.
4. **No repeated mappings.** Commands that behave identically in every application are implemented once. Only genuine per-OS variants appear in platform tables.
5. **Security preserved.** Every new key name must be added to `KeyboardAdapter._SAFE_KEY_NAMES`; `pyautogui.scroll()` is gated behind a new validated `KeyboardAdapter.scroll()` method.

---

## Phase 1 — Configuration Foundation

**Goal:** Establish the desktop command vocabulary and per-use-case inference tuning without touching any Python code.

### 1.1 New Command Map

Create `configs/gestures/command_map_desktop.yaml`:

```yaml
version: v2
# Application-agnostic logical commands.
gesture_commands:
  swipe_up: scroll_forward # universal: advance page in any focused app
  swipe_down: scroll_back # universal: go back in any focused app
  swipe_right: navigate_forward # next workspace or next tab (see navigation_mode)
  swipe_left: navigate_back # previous workspace or previous tab
  zoom_in: zoom_in_view # Ctrl+scroll up — universal view zoom in
  zoom_out: zoom_out_view # Ctrl+scroll down — universal view zoom out
  open_palm_hold: mission_control # OS overview mode toggle
  unknown: no_action
```

The existing `command_map_ipn.yaml` is **not modified**.

### 1.2 Desktop Inference Config

Create `configs/inference_desktop.yaml` with timing tuned for desk interaction:

```yaml
window_length: 24
window_step: 3
min_valid_ratio: 0.8
idle_timeout_ms: 60000 # 1 min; 5-min timeout is excessive for desk use
command_cooldown_ms: 800 # slightly tighter than presentation config (1000 ms)
edge_trigger_enabled: true
gaze:
  enabled: true
  yaw_threshold_deg: 30
  pitch_threshold_deg: 25
  arm_ms: 500
  grace_ms: 3000 # shorter grace than presentation (5000 ms)
  sample_interval_frames: 3
```

### 1.3 AppConfig Schema Extension

**File:** `src/gesture_control/utils/schema_validation.py`

Add two optional fields to `AppConfig`:

```python
class AppConfig(BaseModel):
    ...
    # Selects the active gesture-to-command map filename (without path prefix).
    # Defaults to the IPN presentation map for backwards compatibility.
    command_map: str = "command_map_ipn.yaml"
    # Controls horizontal swipe destination: "workspace" → OS virtual desktop
    # shortcuts; "tabs" → Ctrl+Tab / Ctrl+Shift+Tab (browser/editor tabs).
    navigation_mode: str = Field(default="workspace", pattern="^(workspace|tabs)$")
```

Update `configs/app.yaml` to expose the new fields with their defaults:

```yaml
# existing fields unchanged ...
command_map: command_map_ipn.yaml    # switch to command_map_desktop.yaml for desktop mode
navigation_mode: workspace           # switch to "tabs" for tab-based navigation
```

### 1.4 Bootstrap Command Map Fix

**File:** `src/gesture_control/app/bootstrap.py`

The command map path is hardcoded in `build_container()`. Replace the hardcoded filename with the `app.command_map` config field:

```python
# Current (hardcoded — must be changed):
command_map = _load_command_map(config_root / "gestures" / "command_map_ipn.yaml")

# Required fix:
command_map = _load_command_map(config_root / "gestures" / config.app.command_map)
```

This change depends on `AppConfig.command_map` being defined (§1.3) and must be applied in the same commit. Without it, the desktop command map (`command_map_desktop.yaml`) is never loaded regardless of what `app.yaml` specifies.

### Phase 1 Tests

- Unit test: `AppConfig` accepts `command_map` and `navigation_mode` fields; rejects unknown values.
- Unit test: `command_map_desktop.yaml` loads without schema errors and contains `mission_control` as a command value.
- Unit test: `inference_desktop.yaml` loads and validates against `InferenceConfig`.

### Phase 1 Gate

All existing tests pass unchanged. New config files load without errors.

---

## Phase 2 — Desktop Adapter

**Goal:** Implement a single `DesktopAdapter` covering all desktop use cases. Scroll is universal via OS scroll injection; navigation and overview use per-platform shortcut tables resolved at construction time.

### 2.1 `KeyboardAdapter` Extensions

**File:** `src/gesture_control/infrastructure/dispatch/app_adapters/keyboard_adapter.py`

**a) New `scroll()` method** — gated scroll injection to keep all `pyautogui` calls in one place:

```python
def scroll(self, clicks: int) -> None:
    """Inject a native OS scroll event at the current cursor position.

    Positive clicks scroll up; negative scroll down.  Clamped to [-20, 20]
    to prevent runaway scrolls from rapid-fire command delivery.
    """
    if pyautogui is None:
        LOGGER.warning("pyautogui not available — scroll skipped")
        return
    pyautogui.scroll(max(-20, min(20, clicks)))
```

**b) New `zoom()` method** — Ctrl+scroll injection for application-agnostic view zoom:

```python
# Zoom modifier is platform-specific.  On macOS, Ctrl+scroll activates the
# OS Accessibility screen magnifier — the wrong target.  Cmd+scroll triggers
# in-app zoom in all major browsers, PDF viewers, and editors on macOS.
_ZOOM_MODIFIER: str = "command" if sys.platform == "darwin" else "ctrl"

def zoom(self, clicks: int) -> None:
    """Inject a modifier+scroll zoom event at the current cursor position.

    Uses Cmd+scroll on macOS and Ctrl+scroll on Windows / Linux to trigger
    in-application view zoom without activating OS-level zoom overlays.
    Positive clicks zoom in; negative zoom out.  Clamped to [-5, 5] to
    prevent runaway zoom from rapid-fire command delivery.
    """
    if pyautogui is None:
        LOGGER.warning("pyautogui not available — zoom skipped")
        return
    clamped = max(-5, min(5, clicks))
    pyautogui.keyDown(_ZOOM_MODIFIER)
    try:
        pyautogui.scroll(clamped)
    finally:
        # Always release the modifier — prevents a stuck Ctrl/Cmd key if
        # pyautogui.scroll() raises (e.g. display server unavailable).
        pyautogui.keyUp(_ZOOM_MODIFIER)
```

> **Security note:** `zoom()` accepts a plain `int`. `_ZOOM_MODIFIER` is a module-level constant resolved from `sys.platform` at import time — not user-supplied. `"ctrl"` is already in `_SAFE_KEY_NAMES`; `"command"` must be added (see §2.1c). The `try/finally` guarantees the modifier is always released, preventing a stuck key if `scroll()` raises.

**c) Allowlist extension** — add only keys used in the shortcut tables below:

```python
_SAFE_KEY_NAMES: frozenset[str] = frozenset({
    # existing keys ...
    # New additions for DesktopAdapter:
    "win",      # Windows workspace: Ctrl+Win+Arrow; Windows overview: Win+Tab
    "super",    # Linux overview: Super+A (GNOME), Super+W (KDE)
    "command",  # macOS zoom: Cmd+scroll (in-app zoom, not OS accessibility zoom)
})
```

> **Security note:** `scroll()` and `zoom()` accept plain `int` values; no user-supplied strings reach `pyautogui`. The hotkey path continues to validate every key name against `_SAFE_KEY_NAMES`.

### 2.2 `DesktopAdapter`

Create `src/gesture_control/infrastructure/dispatch/app_adapters/desktop_adapter.py`:

```python
_TARGET_APP = "desktop"

# Navigation shortcuts keyed by (sys.platform, navigation_mode).
_NAV_SHORTCUTS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("darwin",  "workspace"): {"navigate_forward": ("ctrl", "right"), "navigate_back": ("ctrl", "left")},
    ("win32",   "workspace"): {"navigate_forward": ("ctrl", "win", "right"), "navigate_back": ("ctrl", "win", "left")},
    ("linux",   "workspace"): {"navigate_forward": ("ctrl", "alt", "right"), "navigate_back": ("ctrl", "alt", "left")},
    ("darwin",  "tabs"):      {"navigate_forward": ("ctrl", "tab"), "navigate_back": ("ctrl", "shift", "tab")},
    ("win32",   "tabs"):      {"navigate_forward": ("ctrl", "tab"), "navigate_back": ("ctrl", "shift", "tab")},
    ("linux",   "tabs"):      {"navigate_forward": ("ctrl", "tab"), "navigate_back": ("ctrl", "shift", "tab")},
}

# OS overview (Mission Control / Task View / Activities) shortcuts.
_OVERVIEW_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "darwin": ("ctrl", "up"),   # Mission Control  (F3 is equivalent)
    "win32":  ("win", "tab"),   # Task View
    "linux":  ("super", "a"),   # GNOME Activities (Super+W for KDE Plasma Overview)
}

_SCROLL_CLICKS = 5   # configurable; units are pyautogui scroll "clicks"
_ZOOM_CLICKS  = 3   # configurable; smaller range than scroll to avoid over-zoom
```

The adapter's `execute()` method dispatches on `command_id`:

- `scroll_forward` / `scroll_back` → `keyboard.scroll(±_SCROLL_CLICKS)`
- `zoom_in_view` / `zoom_out_view` → `keyboard.zoom(±_ZOOM_CLICKS)`
- `navigate_forward` / `navigate_back` → `keyboard.press_hotkey(*_NAV_SHORTCUTS[(platform, navigation_mode)][command_id])`
- `mission_control` → `keyboard.press_hotkey(*_OVERVIEW_SHORTCUTS[platform])`
- `no_action` → returns `DispatchResult(success=True, ...)` immediately

The `navigation_mode` and `platform` values are injected at construction time (defaulting to `config.app.navigation_mode` and `sys.platform`), so the adapter itself contains no runtime branching.

**Platform normalisation and validation at `__init__` time:**

```python
_KNOWN_PLATFORMS = frozenset({"darwin", "win32", "linux"})

# Normalise Linux variants (e.g. "linux2" on older CPython builds) to "linux".
if platform.startswith("linux"):
    platform = "linux"
if platform not in _KNOWN_PLATFORMS:
    raise ValueError(
        f"DesktopAdapter: unsupported platform {platform!r}. "
        "Supported: darwin, win32, linux."
    )
self._platform = platform
```

This raises a descriptive `ValueError` at construction rather than a bare `KeyError` at dispatch time when `_NAV_SHORTCUTS[(platform, navigation_mode)]` is accessed on an unrecognised platform.

### 2.3 Allowed Commands

The `CommandDispatcher` for the desktop adapter receives a dedicated allowlist built from `frozenset(command_map.values())` in `bootstrap.py` — the same mechanism used for all adapters. The constant in `desktop_adapter.py` is a **documentation artefact** that declares the expected command vocabulary for readability, but it is not the runtime-enforced set:

```python
# Documentation constant — expected values from command_map_desktop.yaml.
# Runtime allowlist is built in bootstrap.py: frozenset(command_map.values()).
_DESKTOP_ALLOWED = frozenset({
    "scroll_forward", "scroll_back",
    "zoom_in_view", "zoom_out_view",
    "navigate_forward", "navigate_back",
    "mission_control",
    "no_action",
})
```

Presentation-domain command IDs (`next_slide`, `start_presentation`, etc.) are silently rejected.

### 2.4 Bootstrap Integration

**File:** `src/gesture_control/app/bootstrap.py`

Add one branch to `_build_dispatcher()`:

```python
elif active_app == "desktop":
    adapter = DesktopAdapter(
        keyboard=keyboard,
        navigation_mode=config.app.navigation_mode,
    )
```

### Phase 2 Tests

| Test file                                      | Scope                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/test_desktop_adapter.py`           | All 8 command IDs; per-platform nav shortcuts; `sys.platform` patching; `navigation_mode` switching; zoom sign convention |
| `tests/unit/test_keyboard_adapter.py` (extend) | `scroll()` clamps to [-20, 20]; `zoom()` clamps to [-5, 5] and sends `keyDown`/`keyUp` with `"command"` on macOS / `"ctrl"` on other platforms; `try/finally` releases modifier on `scroll()` failure; new key names pass allowlist |

### Phase 2 Gate

All unit tests pass. `python scripts/run_realtime.py --config configs --active-app desktop` starts without error via the existing smoke-test fixture.

---

## Phase 3 — Navigation Mode and Scroll Tuning

**Goal:** Expose user-facing controls for navigation target and scroll sensitivity. No new infrastructure is required — Phase 2 already wires `navigation_mode` into `DesktopAdapter` at construction; this phase adds the remaining config knobs and validates runtime behaviour.

### 3.1 `DesktopConfig` Schema Block

**File:** `src/gesture_control/utils/schema_validation.py`

Add `DesktopConfig` as an optional nested block inside **`AppConfig`** — not `RuntimeConfig`. `load_runtime_config()` constructs `AppConfig` directly from `{dir}/app.yaml`; placing `DesktopConfig` here means `configs/desktop/app.yaml` loads it via the existing YAML path without any loader changes. Adding it to `RuntimeConfig` would have no YAML loading path.

```python
class DesktopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Number of pyautogui scroll clicks per scroll gesture event.
    # Increase for fast scrolling (long terminal output); decrease for
    # precision reading (dense code or research papers).
    scroll_clicks: int = Field(default=5, ge=1, le=20)
    # Number of modifier+scroll steps per zoom gesture event.
    # Upper bound matches KeyboardAdapter.zoom() internal clamp of [-5, 5].
    zoom_clicks: int = Field(default=3, ge=1, le=5)
    # Linux-specific: desktop environment hint for mission_control shortcut.
    # "gnome" → Super+A;  "kde" → Super+W;  "other" → bare Super key.
    linux_de: str = Field(default="gnome", pattern="^(gnome|kde|other)$")

class AppConfig(BaseModel):
    ...
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
```

Update `DesktopAdapter` to read `scroll_clicks` and `zoom_clicks` from config at construction (replaces the module-level `_SCROLL_CLICKS` and `_ZOOM_CLICKS` constants).

For Linux, refine `_OVERVIEW_SHORTCUTS` to branch on `linux_de`:

```python
_OVERVIEW_SHORTCUTS_LINUX: dict[str, tuple[str, ...]] = {
    "gnome":  ("super", "a"),   # GNOME Activities overview
    "kde":    ("super", "w"),   # KDE Plasma Overview
    "other":  ("super",),       # fallback: bare Super key
}
```

### 3.2 `configs/desktop/` Profile Directory

Create a `configs/desktop/` directory — two files are required. `load_runtime_config()` accepts a **directory path** and reads `{dir}/app.yaml` and `{dir}/inference.yaml` by convention. It does not accept a standalone YAML file path, so a flat `configs/app_desktop.yaml` will not be discovered.

**`configs/desktop/app.yaml`:**

```yaml
# Desktop interaction profile — invoke with: --config configs/desktop
app_name: gesture-control-desktop
camera_index: 0
frame_width: 1280
frame_height: 720
target_fps: 30
active_app: desktop
command_map: command_map_desktop.yaml
navigation_mode: workspace  # switch to "tabs" for browser/editor tab navigation
mirror_display: true

desktop:
  scroll_clicks: 5
  zoom_clicks: 3
  linux_de: gnome  # linux only: gnome | kde | other
```

**`configs/desktop/inference.yaml`** — desktop-tuned timing (same content as Phase 1.2’s `configs/inference_desktop.yaml`; the loader reads `inference.yaml` by name from the config directory):

```yaml
window_length: 24
window_step: 3
min_valid_ratio: 0.8
idle_timeout_ms: 60000
command_cooldown_ms: 800
edge_trigger_enabled: true
gaze:
  enabled: true
  yaw_threshold_deg: 30
  pitch_threshold_deg: 25
  arm_ms: 500
  grace_ms: 3000
  sample_interval_frames: 3
```

> `configs/inference_desktop.yaml` (from Phase 1.2) is kept as a reference artefact but is not loaded at runtime. The active desktop inference config is `configs/desktop/inference.yaml`.

### Phase 3 Tests

- Unit test: `DesktopConfig` validates `scroll_clicks` bounds (`ge=1, le=20`), `zoom_clicks` bounds (`ge=1, le=5`), and `linux_de` pattern.
- Unit test: `DesktopAdapter` passes `scroll_clicks` to `KeyboardAdapter.scroll()` and `zoom_clicks` to `KeyboardAdapter.zoom()`; verified via mock.
- Unit test: `DesktopAdapter` raises `ValueError` at construction for an unsupported platform string.
- Unit test: Linux `mission_control` shortcut resolves correctly for `gnome`, `kde`, and `other`.

### Phase 3 Gate

`python scripts/run_realtime.py --config configs/app_desktop.yaml` starts without error. `DesktopAdapter.execute("mission_control")` calls the correct OS shortcut for the configured `linux_de`.

---

## Phase 4 — Feedback Improvements

**Goal:** Surface the active mode and pending command in the overlay so users build accurate mental models of what the system will do.

### 4.1 `RuntimeState` Extension

**File:** `src/gesture_control/domain/entities.py`

Add one field to `RuntimeState`:

```python
@dataclass(slots=True)
class RuntimeState:
    mode: str
    last_label: str = "unknown"
    last_confidence: float = 0.0
    gaze_permitted: bool = True
    active_mode: str = "unknown"    # ← new: e.g. "desktop·workspace" or "desktop·tabs"
    info: dict[str, Any] = field(default_factory=dict)
```

### 4.2 `OverlayRenderer` Updates

**File:** `src/gesture_control/infrastructure/feedback/overlay_renderer.py`

Three targeted changes to the existing `render()` method:

**a) Mode badge** — top-left corner label showing the active navigation mode and adapter:

```
[desktop · workspace]   open_palm_hold  → mission_control  ████████░░  0.91
```

**b) Logical command preview** — render the _logical command_ that would fire given `last_label`, not just the raw gesture label. Requires passing `command_map: dict[str, str]` to `OverlayRenderer.__init__()` so it can look up `open_palm_hold → mission_control`.

**c) State hue ring** — replace subtle border with a coloured ring around the gesture label background:

- `IDLE` → grey
- `ARMED` → green
- `COOLDOWN` → amber

### 4.3 Bootstrap Wiring

Set `active_mode` in `RuntimeState` at bootstrap time from `config.app.active_app` and `config.app.navigation_mode` (e.g. `f"{active_app}·{navigation_mode}"`). The runtime loop reads this field from state without modifications to the dispatch path.

### Phase 4 Tests

- Unit test: `OverlayRenderer` renders mode badge when `active_mode != "unknown"`.
- Unit test: `OverlayRenderer` renders logical command label using provided `command_map`.

### Phase 4 Gate

The overlay displays `[desktop · workspace]  → mission_control` in a test frame rendered against a synthetic `RuntimeState`.

---

## Dependency Map

```
Phase 1 (Config + command map)
    ↓
Phase 2 (DesktopAdapter + KeyboardAdapter.scroll + KeyboardAdapter.zoom)
    ↓
Phase 3 (DesktopConfig scroll_clicks + linux_de)
    ↓
Phase 4 (Overlay: active_mode badge + logical command)
```

---

## File Change Summary

| Phase | Files created                                                                 | Files modified                                                                          |
| ----- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1     | `configs/gestures/command_map_desktop.yaml`, `configs/inference_desktop.yaml` | `configs/app.yaml`, `schema_validation.py`, `config_loader.py`                          |
| 2     | `app_adapters/desktop_adapter.py`                                             | `keyboard_adapter.py`, `bootstrap.py`                                                   |
| 3     | `configs/app_desktop.yaml`                                                    | `schema_validation.py` (`DesktopConfig` block), `desktop_adapter.py`                    |
| 4     | —                                                                             | `domain/entities.py`, `infrastructure/feedback/overlay_renderer.py`, `app/bootstrap.py` |

**Total core pipeline files modified: 0** (feature engineering, LSTM, stabiliser, activation controller, gaze detector are untouched across all phases).

**Adapter count: 1** (was 3 in the original plan). No per-application adapter; no OS context detection infrastructure.

---

**Document:** Desktop Interaction Expansion Roadmap  
**Suite:** `docs/development/`  
**Last Updated:** 2026-04-19
