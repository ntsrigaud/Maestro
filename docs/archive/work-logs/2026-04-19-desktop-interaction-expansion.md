# Work Log — Desktop Interaction Expansion

**Date:** 2026-04-19  
**Branch:** `feat/desktop-interaction-expansion`  
**Author:** @ntsrigaud

---

## Context

The gesture control system was originally designed for presentation and meeting control
with the user standing at lectern distance (1.5–3 m). A systematic analysis confirmed
that the geometry is structurally out of distribution for the IPN Hand training data:
MediaPipe landmark quality collapses at that range, several feature blocks degrade, and
the activation timing is mismatched with the valid-ratio filter. The production model at
desk distance (Macro-F1 = 0.96) is already adequate for a broader desktop interaction
vocabulary. This feature track reframes the system entirely around desk-distance
universal desktop control, implements the new adapter and configuration layer, removes
all presentation-domain infrastructure, and hardens the bootstrap against silent
misconfiguration.

The work was delivered across four sub-branches:

1. `docs/desktop-interaction-expansion` — analysis and roadmap
2. `build/configuration-foundation` — schema extensions and config artefacts
3. `fix/centralize-desktop-configuration` — adapter consolidation and swipe correction
4. `refactor/pipeline-optimization` — dead-code removal and fail-fast hardening

---

## Deliverables

### Documentation (`docs/desktop-interaction-expansion`)

**`docs/development/desktop-interaction-analysis.md`** (new, 137 lines)

Five root-cause evidence items documenting why the presentation scenario is unreliable:
dataset distribution mismatch (IPN at 50–80 cm; presenter at 150–300 cm), MediaPipe
quality collapse below ~80×80 px, `fingertip_spread` compression at distance,
`joint_angles` depth degradation, and `command_cooldown_ms` / `edge_trigger` mismatch.
Four desk-distance interaction scenarios defined: web browsing, document review, media
control, and OS workspace navigation.

**`docs/development/desktop-interaction-roadmap.md`** (new, 492 lines)

Four-phase specification: Phase 1 — config schema extension; Phase 2 — `DesktopAdapter`
implementation; Phase 3 — `DesktopConfig` schema and desktop profile directory; Phase 4
— overlay enrichment. Includes `command_map_desktop.yaml` vocabulary spec, platform zoom
modifier rationale (Cmd on macOS / Ctrl on Linux/Windows), `CommandDispatcher` allowlist
mechanism, and bootstrap wiring details.

---

### Configuration Foundation (`build/configuration-foundation`)

**`configs/gestures/command_map_desktop.yaml`** (new)

Universal 8-gesture command map: `swipe_right → navigate_forward`,
`swipe_left → navigate_back`, `swipe_up → scroll_back`, `swipe_down → scroll_forward`,
`zoom_in → zoom_in_view`, `zoom_out → zoom_out_view`,
`open_palm_hold → mission_control`, `unknown → no_action`.

**`configs/inference_desktop.yaml`** (new, reference artefact)

Desktop-optimised inference timing profile: `window_length: 18`, `window_step: 3`,
`min_valid_ratio: 0.7`, `command_cooldown_ms: 600`.

**`src/gesture_control/utils/schema_validation.py`**

`AppConfig` gains `command_map: str` (default `command_map_desktop.yaml`) and
`navigation_mode: str` (default `workspace`, validated `^(workspace|tabs)$`).
`DesktopConfig` model added: `scroll_clicks`, `zoom_clicks`, `linux_de`. `AppConfig.desktop`
field wired to `DesktopConfig`.

**`src/gesture_control/app/bootstrap.py`**

`_load_command_map()` added; bootstrap reads `command_map` from config and passes the
loaded mapping to `CommandDispatcher` as the runtime allowlist. `_build_dispatcher()`
wired to consume `DesktopConfig` scroll/zoom parameters.

---

### DesktopAdapter and Overlay Enrichment

**`src/gesture_control/infrastructure/dispatch/app_adapters/desktop_adapter.py`**
(new, 202 lines added net)

Handles 8 command IDs via `KeyboardAdapter`. Platform-normalised zoom modifier (Cmd on
macOS, Ctrl on Linux/Windows). Scroll clamped to [-20, 20] clicks; zoom clamped to
[-5, 5]. `mission_control` invokes Ctrl+Up on macOS / Super+S on Linux /
Win+Tab on Windows. `navigate_forward/back` routes to workspace (`Ctrl+Right/Left` on
macOS / `Ctrl+Win+Right/Left` on Linux) or tab (`Ctrl+Tab` / `Ctrl+Shift+Tab`) depending
on `navigation_mode`. `no_action` is a no-op.

**`src/gesture_control/infrastructure/dispatch/app_adapters/keyboard_adapter.py`**

`scroll()` and `zoom()` methods added. `zoom()` accepts a mandatory `modifier: str`
parameter (platform selection is the caller's responsibility). `_SAFE_KEY_NAMES`
allowlist extended for modifier keys.

**`src/gesture_control/infrastructure/feedback/overlay_renderer.py`**

Mode badge `[desktop·workspace]` rendered at frame top. Prediction ring: coloured
background rectangle (dark green in ARMED, dark amber in COOLDOWN, dark grey otherwise)
drawn behind the prediction line. Prediction line extended to show the logical command
mapping: `pred=swipe_right → navigate_forward conf=0.94`. `command_map` parameter added
to `__init__`. `active_mode` read from `RuntimeState`.

**`src/gesture_control/domain/entities.py`** — `active_mode: str = "unknown"` added to
`RuntimeState`.

**`src/gesture_control/app/runtime.py`** — `self._active_mode` pre-computed once at
init as `f"desktop·{nav_mode}"` and written to `RuntimeState.active_mode` in the hot
loop.

---

### Consolidation and Hardening

**`fix/centralize-desktop-configuration`**

- `PowerPointAdapter` (114 lines) and `ZoomAdapter` (78 lines) deleted.
- `command_map_ipn.yaml` and `command_map_powerpoint.yaml` deleted.
- `configs/app.yaml`: `active_app` removed; `command_map: command_map_desktop.yaml` as
  default.
- Vertical swipe direction corrected: `swipe_up → scroll_back`,
  `swipe_down → scroll_forward` (natural/touchpad scroll convention).
- `_build_dispatcher()` removes `active_app` branching; unconditionally constructs
  `DesktopAdapter`.

**`refactor/pipeline-optimization`**

- `CommandID(StrEnum)` deleted from `domain/enums.py` (6 presentation-domain members).
- `KeyboardCommandDispatcher` deleted (67 lines, hardcoded PowerPoint/Zoom shortcut
  tables).
- `_DESKTOP_ALLOWED` documentation constant removed from `desktop_adapter.py`.
- `active_app: str` field removed from `AppConfig` and all config files.
- `_ZOOM_MODIFIER` module-level constant removed from `keyboard_adapter.py`;
  `zoom()` signature changed to require `modifier: str`.
- `_load_command_map` now raises `FileNotFoundError` / `ValueError` on missing or
  malformed files (previously fell back to IPN defaults silently).

---

### Architecture Documentation

`docs/architecture/system-overview.md`, `logical-view.md`, `development-view.md` — all
updated to reflect universal desktop interaction framing: introduction, system objective,
gesture vocabulary table, domain layer descriptions, component table, and configuration
table updated throughout.

---

## Testing

**New test suites:**

- `tests/unit/test_desktop_adapter.py` (271 lines) — 8 command handler tests,
  platform normalization, navigation mode routing, scroll/zoom clamping.
- `tests/unit/test_overlay_renderer.py` (152 lines) — mode badge, prediction ring
  colour, logical command line, mirror mode, landmarks passthrough.
- `tests/unit/test_phase1_config.py` (344 lines) — 5 test classes covering
  `AppConfig` new fields, `command_map_desktop.yaml` structure, `inference_desktop.yaml`
  reference, bootstrap command map resolution, and `DesktopConfig` schema.

**Updated test suites:**

- `tests/unit/test_adapters.py` — `zoom()` calls updated to mandatory `modifier=` kwarg;
  `TestPowerPointAdapter` / `TestZoomAdapter` classes removed.
- `tests/integration/test_gesture_dispatch_pipeline.py` — migrated from
  `PowerPointAdapter` to `DesktopAdapter`; command IDs updated to desktop vocabulary;
  hotkey assertions updated.
- `tests/unit/test_checkpoint_loader.py` — `active_app` removed from fixture dict.
- `tests/unit/test_phase1_config.py` — `TestCommandMapVsAdapterCoverage` added: loads
  `command_map_desktop.yaml` at test time and asserts every command value has a handler
  in `DesktopAdapter`.

---

## Outcome

The system is reframed entirely around desk-distance universal desktop interaction. The
dispatch layer is fully config-driven with no hardcoded vocabulary, no presentation
application logic, no silent fallback, and no `active_app` indirection. The overlay
provides immediate mode-state and command-preview feedback. 76 files changed:
3,202 insertions, 2,723 deletions (net +479 lines, dominated by new analysis docs,
adapter implementation, and test suites).
