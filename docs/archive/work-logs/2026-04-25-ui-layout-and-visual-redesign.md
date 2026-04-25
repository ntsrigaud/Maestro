# Work Log — UI Layout and Visual Redesign: Two-Pane Layout, Grouped Telemetry, and Asset Documentation

**Date:** 2026-04-25  
**Branch:** `refactor/ui-layout-and-visual-redesign`  
**Author:** @ntsrigaud

---

## Context

The initial PySide6 control surface established the runtime lifecycle contract and panel structure but was visually minimal and used a flat single-column layout that did not scale to the number of controls present. The `StatusMonitor` displayed telemetry fields without grouping or visual hierarchy, and `LauncherPanel` used generic button labels and no status indication. This branch redesigns the three structural UI components — `MainWindow`, `StatusMonitor`, and `LauncherPanel` — to produce a more coherent visual design and a layout that clearly separates primary controls from secondary content, without changing the underlying signal-slot architecture or the runtime interface.

---

## Scope of Work

### UI Refactoring

- Redesigned `MainWindow` with a `QSplitter`-based two-pane layout: a fixed 240 px sidebar housing app identity, the `Control` group box, and `StatusMonitor`; and an expanding content pane housing the Camera Preview group box, a tabbed Settings editor, and the Session Log.
- Introduced a full application stylesheet in `MainWindow` using an 8-shade neutral colour scale with an indigo accent, spacing on a 4 px grid, and three type-size steps (11 / 12 / 13 / 15 px). Stylesheet rules are expressed as named object selectors to avoid cascade coupling.
- Redesigned `StatusMonitor` into two `QGroupBox` sections — "Runtime" (Mode, Gaze, Gesture, Confidence progress bar) and "Performance" (FPS, Latency, Valid ratio) — using `QGridLayout` with two columns for key/value pairing.
- Updated `LauncherPanel` to display a status chip (coloured dot + text label) above the primary action button. Renamed button labels to `Start Session` / `Stop Session` / `Starting…` / `Retry` to match the explicit four-state machine. Preview toggle label updated to `Show camera preview`.

### Documentation

- Created `docs/assets/` with a `screenshots/` subdirectory containing three
  representative UI captures: `ui-idle.png`, `ui-running-preview.png`, and
  `ui-running-monitor.png`.
- Added `docs/assets/README.md` defining screenshot naming conventions, capture
  guidelines (resolution, content policy, format, file-size targets), Markdown
  reference patterns, and an update workflow.
- Updated `docs/architecture/ui-view.md` to reflect the two-pane splitter layout,
  including an updated layout overview and diagram.
- Added the live camera preview screenshot to `README.md`.

### Testing

- Updated button text assertions in `tests/unit/test_ui_launcher_panel.py` to match
  the renamed state labels (`Start Session`, `Stop Session`, `Starting…`, `Retry`).
- Updated attribute assertions in `tests/unit/test_ui_status_monitor.py` to reference
  the redesigned widget structure (`_mode_val`, `_gaze_val`, `_fps_val`,
  `_confidence_bar`).

### Style

- Applied Black formatter across modified Python source and test modules.
- Applied Prettier formatting to updated Markdown documentation files.

---

## Technical Changes

**`src/maestro/app/ui/main_window.py`** replaces the previous flat vertical layout with a `QSplitter(Qt.Orientation.Horizontal)` composition. The sidebar `QWidget` (object name `"sidebar"`) is fixed at 240 px via `setFixedWidth`; `setChildrenCollapsible(False)` and `setStretchFactor` prevent accidental collapse. A self-contained `_STYLESHEET` string is applied once at `MainWindow.__init__` and targets widgets by object name (`#sidebar`, `#primaryButton`, `#confidenceBar`, `#sessionLog`, etc.), keeping style rules isolated from layout code. No changes were made to signal wiring or the `RuntimeController` interface.

**`src/maestro/app/ui/status_monitor.StatusMonitor`** now lays out telemetry in two `QGroupBox` sections using `QGridLayout`. The Confidence field uses a `QProgressBar` (object name `"confidenceBar"`, range 0–100) instead of a plain label, giving an immediate proportional reading. Gaze state updates use object-name switching (`"gazeActive"` / `"gazeAbsent"`) to trigger stylesheet colour changes, followed by explicit `unpolish`/`polish` calls to force re-evaluation. The `on_state` slot signature is unchanged.

**`src/maestro/app/ui/launcher_panel.LauncherPanel`** replaces the bare button with a status chip row (coloured 10×10 `QLabel` dot + `"statusChipLabel"` text) above the `QPushButton`. The `_LABELS` and `_CHIP_TEXT` class-level dicts drive both the chip text and the button label from the same four-state key (`idle`, `building`, `running`, `error`), ensuring consistency. The `show_window` property and all signal definitions are unchanged.

**`docs/assets/README.md`** establishes a documented convention for screenshot lifecycle: naming pattern (`<scope>-<state>[--<variant>].png`), capture resolution (1× pixel density, ~1400 px window width), format requirement (PNG, ≤ 500 KB via `pngcrush` or `oxipng`), and Markdown reference examples relative to both `docs/architecture/` and the repository root.

---

## Testing and Validation

- `test_ui_launcher_panel.py` updated: all four state-transition assertions now use the renamed labels; signal-emission and disabled-state tests are unaffected.
- `test_ui_status_monitor.py` updated: assertions reference redesigned widget attributes; `on_state` behaviour tests pass without structural change.
- All other UI unit suites (`test_ui_config_panel.py`, `test_ui_model_panel.py`, `test_ui_runtime_controller.py`, `test_ui_runtime_on_state.py`) required no modification, confirming that the redesign did not alter signal contracts or controller interfaces.
- Full unit suite passes; no suppressed tests introduced.

---

## Outcome

The graphical control surface now uses a structured two-pane layout that separates primary runtime controls from secondary configuration and preview content. Telemetry is presented in semantically grouped cards with a confidence progress bar rather than raw text. Button labels and status chip indicators communicate lifecycle state unambiguously. A dedicated asset directory and guide establish a reproducible workflow for capturing and referencing UI screenshots in documentation. No breaking changes were introduced to the runtime interface, signal architecture, or CLI execution path.
