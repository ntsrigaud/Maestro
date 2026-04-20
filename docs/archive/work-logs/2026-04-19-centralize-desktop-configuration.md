# Work Log — Centralize Desktop Configuration and Remove Application-Specific Adapters

**Date:** 2026-04-19  
**Branch:** `feat/centralize-desktop-configuration`  
**Author:** @ntsrigaud

---

## Context

The application had accumulated two presentation-domain adapters (`PowerPointAdapter`,
`ZoomAdapter`) and two command maps (`command_map_ipn.yaml`,
`command_map_powerpoint.yaml`) that were superseded by the universal `DesktopAdapter`
introduced in the desktop interaction expansion. The bootstrap wired adapters via an
`active_app` dispatch table that retained `PowerPointAdapter` as the silent default,
meaning any misconfigured or unknown `active_app` value silently fell back to the
presentation vocabulary. This branch removes all presentation-domain infrastructure,
promotes the desktop profile as the application default, and corrects the vertical swipe
direction convention.

---

## Technical Changes

### `configs/gestures/` — Command Map Consolidation

`command_map_ipn.yaml` and `command_map_powerpoint.yaml` deleted. `command_map_desktop.yaml`
is now the sole command map. Vertical swipe mappings corrected to match natural/touchpad
scroll direction:

| Gesture      | Before           | After            |
| ------------ | ---------------- | ---------------- |
| `swipe_up`   | `scroll_forward` | `scroll_back`    |
| `swipe_down` | `scroll_back`    | `scroll_forward` |

`swipe_up → scroll_back` scrolls content upward (natural direction: content tracks the
hand). The corrected semantics are consistent across browsers, PDF viewers, and
presentation software — `scroll_back` in a slideshow advances to the previous slide via
`pyautogui.scroll(+N)`.

### `configs/app.yaml`

`active_app` changed from `"powerpoint"` to `"desktop"`. `command_map` changed from
`"command_map_powerpoint.yaml"` to `"command_map_desktop.yaml"`.

### `src/gesture_control/utils/schema_validation.py`

`AppConfig.command_map` default updated from `"command_map_ipn.yaml"` to
`"command_map_desktop.yaml"`. The comment referencing "backwards compatibility with the
IPN presentation map" removed as it no longer applies.

### `src/gesture_control/app/bootstrap.py`

`_build_dispatcher()` simplified: the `active_app` parameter and the three-branch
`if/elif/else` dispatch table are removed. `DesktopAdapter` is constructed unconditionally.
Imports of `PowerPointAdapter`, `ZoomAdapter`, and `IApplicationAdapter` removed. The
`active_app` argument is no longer passed at the call site.

Before: unknown `active_app` values silently fell back to `PowerPointAdapter`.
After: a single well-defined adapter with no silent fallback path.

### Infrastructure — Deleted Files

- `src/gesture_control/infrastructure/dispatch/app_adapters/powerpoint_adapter.py`
  (114 lines)
- `src/gesture_control/infrastructure/dispatch/app_adapters/zoom_adapter.py` (78 lines)

---

## Testing

**`tests/unit/test_adapters.py`** — `TestPowerPointAdapter` and `TestZoomAdapter` classes
removed (93 lines deleted). Remaining `TestKeyboardAdapter` tests unchanged.

**`tests/integration/test_gesture_dispatch_pipeline.py`** — migrated from
`PowerPointAdapter` to `DesktopAdapter`. Command map updated to the universal desktop
vocabulary. Presentation-domain command ID assertions (`"next_slide"`,
`"previous_slide"`) replaced with desktop IDs (`"navigate_forward"`,
`"navigate_back"`). Hotkey assertion updated: `mock_pya.hotkey("right")` →
`mock_pya.hotkey("ctrl", "right")`.

**`tests/unit/test_phase1_config.py`** — default assertions updated:
`command_map_ipn.yaml` / `command_map_powerpoint.yaml` → `command_map_desktop.yaml`;
vertical swipe direction assertions corrected to match the new mapping
(`swipe_up → scroll_back`, `swipe_down → scroll_forward`).

Black formatting applied across all modified Python files.

---

## Outcome

The presentation-domain adapter layer is fully removed. `DesktopAdapter` is the sole
dispatch target; `command_map_desktop.yaml` is the default command map; and the bootstrap
has no implicit fallback behaviour. The vertical swipe direction convention is corrected
to natural/touchpad semantics. Net change: 208 insertions, 485 deletions.
