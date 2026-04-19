# Work Log — Pipeline Optimization: Remove Deprecated Dispatch Layer

**Date:** 2026-04-19  
**Branch:** `refactor/pipeline-optimization`  
**Author:** @ntsrigaud

---

## Context

Following the desktop interaction consolidation, several pre-desktop infrastructure
artefacts remained in the codebase: the `CommandID` enum (a presentation-domain command
vocabulary), `KeyboardCommandDispatcher` (the original stub dispatcher with hardcoded
PowerPoint/Zoom shortcut tables), the `_DESKTOP_ALLOWED` documentation constant (made
redundant once the allowlist was moved to the loaded command map), the `_ZOOM_MODIFIER`
platform-sniffing constant (implicit platform coupling in `KeyboardAdapter`), and
`active_app` as a schema field and `app.yaml` key (no longer meaningful with a single
adapter). This branch removes all of them and hardens the bootstrap's command map loader
to fail fast on missing files rather than silently substituting IPN defaults.

---

## Technical Changes

### `src/gesture_control/domain/enums.py` — `CommandID` Removed

`CommandID(StrEnum)` deleted (6 members: `NEXT_SLIDE`, `PREVIOUS_SLIDE`, `RAISE_HAND`,
`START_PRESENTATION`, `TOGGLE_POINTER_MODE`, `EXIT_PRESENTATION`). The presentation
vocabulary was never used by `DesktopAdapter`; command routing is string-keyed via the
YAML command map. `enums.py` now contains only `RuntimeMode`.

### `src/gesture_control/infrastructure/dispatch/keyboard_dispatcher.py` — Deleted

`KeyboardCommandDispatcher` (67 lines) deleted. It held hardcoded shortcut tables for
`powerpoint` and `zoom` app contexts and dispatched via `active_app` string lookup.
Superseded in full by `CommandDispatcher` + `DesktopAdapter`.

`dispatch/__init__.py` cleared of the now-dead `KeyboardCommandDispatcher` re-export.

### `src/gesture_control/infrastructure/dispatch/app_adapters/desktop_adapter.py` — `_DESKTOP_ALLOWED` Removed

`_DESKTOP_ALLOWED: frozenset[str]` was documented as a reference constant only —
runtime allowlist enforcement had already moved to `bootstrap.py` (built from the loaded
command map). Removed to avoid the divergence risk of keeping two parallel vocabulary
lists. The canonical set is now `_DESKTOP_ADAPTER_HANDLED` in the test suite
(`test_phase1_config.py`), which asserts all command map values have a corresponding
handler.

### `src/gesture_control/infrastructure/dispatch/app_adapters/keyboard_adapter.py` — `zoom()` Signature Hardened

`_ZOOM_MODIFIER: str = "command" if sys.platform == "darwin" else "ctrl"` removed.
`zoom()` signature changed from `zoom(self, clicks, *, modifier: str | None = None)` to
`zoom(self, clicks, *, modifier: str)`. The modifier is now a mandatory caller-supplied
argument. Platform selection logic moved up to `DesktopAdapter`, which is the only
caller and already knows the platform. `sys` import removed.

This eliminates implicit platform coupling at the `KeyboardAdapter` layer and makes the
modifier an explicit, testable parameter.

### `src/gesture_control/utils/schema_validation.py` — `active_app` Removed

`active_app: str` field removed from `AppConfig`. The field had no default and required
every config file to declare it explicitly; the only valid value post-consolidation was
`"desktop"`. The `desktop` field comment updated: "Desktop-specific fine-tuning for
scroll/zoom sensitivity and Linux DE." (removed stale `active_app != "desktop"` caveat).

### `configs/app.yaml`, `configs/desktop/app.yaml`

`active_app: desktop` removed from both files. Comment on `navigation_mode` updated:
"# Horizontal swipe destination (workspace | tabs)." (dropped the now-false "Relevant
only when active_app is 'desktop'" qualifier).

### `src/gesture_control/app/bootstrap.py` — `_load_command_map` Fail-Fast

IPN default fallback dict removed. The three silent-return branches replaced with
explicit exceptions:

| Condition                   | Before              | After                                       |
| --------------------------- | ------------------- | ------------------------------------------- |
| File absent                 | return IPN defaults | `FileNotFoundError` with diagnostic message |
| Non-dict YAML root          | return IPN defaults | `ValueError`                                |
| Non-dict `gesture_commands` | return IPN defaults | `ValueError`                                |

A missing or malformed command map now fails immediately at startup with a clear message
rather than running silently on a stale vocabulary.

### `src/gesture_control/app/runtime.py`

`self._active_mode` no longer reads `active_app` from config via `getattr`. Hardcoded
to `f"desktop·{nav_mode}"`. The `active_app` getattr fallback (`""`) and the
conditional f-string (`f"{active_app}·{nav_mode}" if active_app else "unknown"`) are
replaced with one unconditional assignment.

---

## Documentation

`docs/architecture/system-overview.md` — Introduction and System Objective updated to
remove presentation/meeting software framing. Gesture vocabulary table updated from the
IPN/presentation command labels to the desktop command vocabulary
(`navigate_forward/back`, `scroll_back/forward`, `zoom_in/out_view`, `mission_control`).
Narrative paragraph updated to describe desktop interaction modes.

`docs/architecture/logical-view.md` — `CommandID` removed from domain layer Mermaid
diagram node and from the Enums table row. Application Adapters row updated:
`PowerPointAdapter`/`ZoomAdapter` → `DesktopAdapter` description.

`docs/architecture/development-view.md` — `enums.py` row: `CommandID` removed from
contents and description. Configuration table: `command_map_ipn.yaml` reference replaced
with `command_map_desktop.yaml`. `AppConfig` description: "active application" removed.

---

## Testing

**`tests/unit/test_adapters.py`** — Three `zoom()` call sites updated to pass
`modifier="ctrl"` explicitly. No logic changes.

**`tests/unit/test_checkpoint_loader.py`** — `"active_app": "powerpoint"` removed from
the test `AppConfig` fixture dict.

**`tests/unit/test_phase1_config.py`** — Multiple changes:

- `active_app` removed from all `_minimal()` helper dicts and from all direct assertions
  (`config.app.active_app == "desktop"` assertions deleted).
- Module docstring updated: "configs/desktop/app.yaml sets active_app=desktop" →
  "configs/desktop/app.yaml sets command_map_desktop.yaml".
- `TestBootstrapCommandMapResolution` gains
  `test_missing_command_map_raises_file_not_found`: asserts `_load_command_map` raises
  `FileNotFoundError` with message matching `"Command map file not found"`.
- New class `TestCommandMapVsAdapterCoverage` (3 lines of logic + `_DESKTOP_ADAPTER_HANDLED`
  frozenset): loads `command_map_desktop.yaml` at test time and asserts every mapped
  command value is present in the known-handled set. This replaces the removed
  `_DESKTOP_ALLOWED` production constant with an active test assertion.

---

## Outcome

All presentation-domain and pre-desktop infrastructure artefacts removed. The dispatch
layer is now fully config-driven with no hardcoded vocabulary, no platform-sniffing in
`KeyboardAdapter`, no silent IPN fallback, and no `active_app` indirection. Net change:
88 insertions, 163 deletions across 16 files.
