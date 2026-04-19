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
  open_palm_hold: mission_control # OS overview mode toggle
  zoom_in: no_action # excluded — unreliable at desk distance
  zoom_out: no_action # excluded — unreliable at desk distance
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
    # "static" = use active_app at startup (current behaviour)
    # "context_aware" = query OS for frontmost window each dispatch cycle
    dispatch_mode: str = Field(default="static", pattern="^(static|context_aware)$")
```

Update `configs/app.yaml` to expose the new fields with their defaults:

```yaml
# existing fields unchanged ...
command_map: command_map_ipn.yaml # switch to command_map_desktop.yaml for desktop mode
dispatch_mode: static # switch to context_aware when Phase 3 is complete
```

### 1.4 Config Loader Update

**File:** `src/gesture_control/utils/config_loader.py`

The command map is currently loaded by name from `configs/gestures/`. Verify the loader reads `app.command_map` to resolve the path, or add that resolution if it reads a hardcoded filename.

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

**b) Allowlist extension** — add only keys used in the shortcut tables below:

```python
_SAFE_KEY_NAMES: frozenset[str] = frozenset({
    # existing keys ...
    # New additions for DesktopAdapter:
    "win",      # Windows workspace: Ctrl+Win+Arrow; Windows overview: Win+Tab
    "super",    # Linux overview: Super+A (GNOME), Super+W (KDE)
})
```

> **Security note:** `scroll()` accepts a plain `int`; no user-supplied strings reach `pyautogui`. The hotkey path continues to validate every key name against `_SAFE_KEY_NAMES`.

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
```

The adapter's `execute()` method dispatches on `command_id`:

- `scroll_forward` / `scroll_back` → `keyboard.scroll(±_SCROLL_CLICKS)`
- `navigate_forward` / `navigate_back` → `keyboard.press_hotkey(*_NAV_SHORTCUTS[(platform, navigation_mode)][command_id])`
- `mission_control` → `keyboard.press_hotkey(*_OVERVIEW_SHORTCUTS[platform])`
- `no_action` → returns `DispatchResult(success=True, ...)` immediately

The `navigation_mode` and `platform` values are injected at construction time (defaulting to `config.app.navigation_mode` and `sys.platform`), so the adapter itself contains no runtime branching.

### 2.3 Allowed Commands

The `CommandDispatcher` for the desktop adapter receives a dedicated allowlist:

```python
_DESKTOP_ALLOWED = frozenset({
    "scroll_forward", "scroll_back",
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

| Test file                                      | Scope                                                                                               |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `tests/unit/test_desktop_adapter.py`           | All 6 command IDs; per-platform nav shortcuts; `sys.platform` patching; `navigation_mode` switching |
| `tests/unit/test_keyboard_adapter.py` (extend) | `scroll()` clamps to [-20, 20]; new key names pass allowlist                                        |

### Phase 2 Gate

All unit tests pass. `python scripts/run_realtime.py --config configs --active-app desktop` starts without error via the existing smoke-test fixture.

---

## Phase 3 — Navigation Mode and Scroll Tuning

**Goal:** Expose user-facing controls for navigation target and scroll sensitivity. No new infrastructure is required — Phase 2 already wires `navigation_mode` into `DesktopAdapter` at construction; this phase adds the remaining config knobs and validates runtime behaviour.

### 3.1 `DesktopConfig` Schema Block

**File:** `src/gesture_control/utils/schema_validation.py`

Add an optional `desktop` block to `RuntimeConfig`:

```python
class DesktopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Number of pyautogui scroll clicks per gesture event.
    # Increase for fast scrolling (e.g. long terminal output); decrease for
    # precision reading (e.g. dense code or research papers).
    scroll_clicks: int = Field(default=5, ge=1, le=20)
    # Linux-specific: desktop environment hint for mission_control shortcut
    # selection. "gnome" uses Super+A; "kde" uses Super+W.
    linux_de: str = Field(default="gnome", pattern="^(gnome|kde|other)$")

class RuntimeConfig(BaseModel):
    ...
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
```

Update `DesktopAdapter` to read `scroll_clicks` from config at construction (replaces the module-level `_SCROLL_CLICKS` constant).

For Linux, refine `_OVERVIEW_SHORTCUTS` to branch on `linux_de`:

```python
_OVERVIEW_SHORTCUTS_LINUX: dict[str, tuple[str, ...]] = {
    "gnome":  ("super", "a"),   # GNOME Activities overview
    "kde":    ("super", "w"),   # KDE Plasma Overview
    "other":  ("super",),       # fallback: bare Super key
}
```

### 3.2 `configs/app_desktop.yaml` Example

Provide a ready-to-use desktop configuration example alongside the existing `app.yaml`:

```yaml
# configs/app_desktop.yaml — desktop interaction profile
app_name: gesture-control-desktop
camera_index: 0
frame_width: 1280
frame_height: 720
target_fps: 30
active_app: desktop
command_map: command_map_desktop.yaml
navigation_mode: workspace # or: tabs
mirror_display: true

desktop:
  scroll_clicks: 5
  linux_de: gnome
```

### Phase 3 Tests

- Unit test: `DesktopConfig` validates `scroll_clicks` bounds and `linux_de` pattern.
- Unit test: `DesktopAdapter` passes `scroll_clicks` to `KeyboardAdapter.scroll()`; verified via mock.
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

## Phase 5 — Dataset and Model Robustness (Future)

**Goal:** Improve model robustness at moderate distances (80–100 cm) and for `zoom_in`/`zoom_out` at desk range, without redesigning the pipeline.

### 5.1 Spatial Scale Augmentation

**File:** `src/gesture_control/training/augmentation.py`

Add a `spatial_scale` parameter to `AugmentationConfig` and a corresponding transform in `GestureAugmenter`:

```python
# Simulate the smaller hand apparent size at moderate distance
# by uniformly scaling all landmarks toward the wrist origin.
scale = uniform(spatial_scale_min, spatial_scale_max)   # e.g. (0.6, 1.0)
landmarks *= scale
```

This augmentation targets the `positions[0:63]` and `velocity[71:134]` blocks and does not require any architecture change.

### 5.2 Desk-Distance Calibration Dataset

Write a lightweight collection script `scripts/collect_desk_dataset.py` that:

1. Runs the inference pipeline in recording mode (no dispatch)
2. Writes labeled JSONL landmark files to `data/raw/desk_collected/`
3. Annotates each sequence with the gesture class via keyboard prompt during capture

Recording protocol: 5 participants × 20 gestures per class × 8 classes = 800 sequences (≈ 30 min per participant at desk distance, 50–80 cm).

These recordings integrate with the existing `build_ipn_dataset.py` pipeline and `GestureDataModule` without code changes.

### 5.3 Fine-Tuning Workflow

The `notebooks/02_model_training.ipynb` notebook is reused as-is. Load the existing `lstm_v1_20260418T023120Z_inference.pt` weights as a starting checkpoint and fine-tune for 10–20 epochs on the combined IPN + desk dataset with a low learning rate (1e-4).

Publish the updated artifact to HF Hub as a new revision via `training/hf_publish.py`. Update `configs/model_source.yaml` to pin the new filename.

### Phase 5 Gate

Spatial scale augmentation: training F1 on the IPN validation split does not decrease by more than 0.02 compared to baseline. Desk-distance test split: all four swipe classes achieve F1 ≥ 0.85.

---

## Dependency Map

```
Phase 1 (Config + command map)
    ↓
Phase 2 (DesktopAdapter + KeyboardAdapter.scroll)
    ↓
Phase 3 (DesktopConfig scroll_clicks + linux_de)
    ↓
Phase 4 (Overlay: active_mode badge + logical command)
    ↓
Phase 5 (Dataset augmentation)  ← independent; can run in parallel after Phase 1
```

---

## File Change Summary

| Phase | Files created                                                                 | Files modified                                                                          |
| ----- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1     | `configs/gestures/command_map_desktop.yaml`, `configs/inference_desktop.yaml` | `configs/app.yaml`, `schema_validation.py`, `config_loader.py`                          |
| 2     | `app_adapters/desktop_adapter.py`                                             | `keyboard_adapter.py`, `bootstrap.py`                                                   |
| 3     | `configs/app_desktop.yaml`                                                    | `schema_validation.py` (`DesktopConfig` block), `desktop_adapter.py`                    |
| 4     | —                                                                             | `domain/entities.py`, `infrastructure/feedback/overlay_renderer.py`, `app/bootstrap.py` |
| 5     | `scripts/collect_desk_dataset.py`                                             | `training/augmentation.py`, `configs/model_source.yaml`                                 |

**Total core pipeline files modified: 0** (feature engineering, LSTM, stabiliser, activation controller, gaze detector are untouched across all phases).

**Adapter count: 1** (was 3 in the original plan). No per-application adapter; no OS context detection infrastructure.

---

**Document:** Desktop Interaction Expansion Roadmap  
**Suite:** `docs/development/`  
**Last Updated:** 2026-04-19
