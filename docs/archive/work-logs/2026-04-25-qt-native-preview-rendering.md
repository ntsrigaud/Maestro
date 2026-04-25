# Work Log — Qt-Native Preview Rendering: OpenCV Display Elimination and Unified Frame Pipeline

**Date:** 2026-04-25  
**Branch:** `feat/Qt-native-preview-rendering`  
**Author:** @ntsrigaud

---

## Context

The previous UI branch introduced a PySide6 control panel and wired `Runtime.on_state`
as a delivery channel for live telemetry, but retained an OpenCV-backed display path
(`cv2.imshow()`) that violated Qt threading rules when the GUI was active. The runtime
loop called `cv2.imshow()` directly from a `QThread` worker, where it could conflict
with the Qt event loop on the main thread. This branch removes that violation by
eliminating all direct OpenCV display calls from the inference runtime and replacing them
with a fully Qt-native frame pipeline in which rendered frames are delivered to display
consumers via `RuntimeState`.

---

## Scope of Work

### Domain and Runtime Refactoring

- Added `preview_frame` field to `RuntimeState` in `src/maestro/domain/entities.py`;
  populated by `Runtime` when `show_window=True` with the rendered canvas produced by
  `OverlayRenderer`, and left `None` otherwise.
- Removed all `cv2.imshow()` and `cv2.waitKey()` calls from
  `src/maestro/app/runtime.py`; frame display is now exclusively the responsibility
  of downstream consumers of `RuntimeState`.
- Decoupled the `show_window` flag from OpenCV: setting `show_window=True` now
  populates `state.preview_frame` and triggers the Qt preview path rather than opening
  a cv2 window.

### Infrastructure — Overlay Renderer

- Added `output_rgb: bool = False` constructor parameter to `OverlayRenderer` in
  `src/maestro/infrastructure/feedback/overlay_renderer.py`.
- When `output_rgb=True`, the renderer applies a single BGR→RGB channel reversal
  (`canvas[..., ::-1]`) after all drawing operations, consolidating colour-space
  conversion at the source rather than repeating it on every display update.

### UI Layer

- Implemented `PreviewWidget` (`QLabel` subclass) in
  `src/maestro/app/ui/preview_widget.py`: a Qt-native camera preview panel that
  receives `RuntimeState` via a queued-connection `on_state` slot, constructs a
  `QImage(Format_RGB888)` directly from the already-RGB frame bytes, and scales the
  result to a maximum display width of 480 px with aspect-ratio preservation. The
  widget starts hidden and becomes visible on the first frame carrying a non-`None`
  `preview_frame`.
- Implemented `CLIPreviewWindow` in `src/maestro/app/ui/cli_preview.py`: a minimal
  `QMainWindow` that pairs `RuntimeController` with `PreviewWidget` for CLI preview
  mode. Keyboard shortcuts Q and Escape stop the runtime gracefully; window close also
  terminates the runtime.
- Integrated `PreviewWidget` into `src/maestro/app/ui/main_window.py` so the full GUI
  control panel and the CLI preview window share a single rendering component.

### Application Entry Point

- Refactored `src/maestro/app/main.py` to fork execution on the `--no-display` flag:
  - With display: boots a `QApplication`, creates `CLIPreviewWindow`, and runs the Qt
    event loop — no OpenCV window or `cv2` import in this path.
  - Headless (`--no-display`): constructs and runs `Runtime` directly with
    `show_window=False`; benchmark output and `--max-frames` remain available in this
    path.

### Style

- Applied Black formatter across modified and touched source modules.
- Removed unused `QEvent` import that produced a linting error.

---

## Technical Changes

**`src/maestro/domain/entities.RuntimeState`** gains a `preview_frame: Any` field
(default `None`). `Runtime.run()` attaches `overlay_frame.frame` to this field
unconditionally when `show_window=True`, in both the gaze-bypassed path and the main
inference path. The field carries a fresh canvas copy produced by `OverlayRenderer`;
consumers need not copy it further.

**`src/maestro/infrastructure/feedback/overlay_renderer.OverlayRenderer`** accepts
`output_rgb: bool = False`. When the flag is set, the final rendered canvas is
converted in-place with `canvas[..., ::-1]` before being wrapped in a new
`FramePacket`. Bootstrap code sets this flag to `True` whenever `show_window=True` so
that `PreviewWidget` can pass frame bytes directly to `QImage.Format_RGB888` without an
additional copy or conversion.

**`src/maestro/app/ui/preview_widget.PreviewWidget`** is a `QLabel` subclass connected
to `RuntimeController.state_updated` via Qt's queued-connection mechanism. The queued
connection guarantees that `on_state()` executes on the main thread regardless of which
thread emits the signal, eliminating the threading violation that motivated this
refactor. Pixmap scaling is applied only when the frame width exceeds `_MAX_DISPLAY_WIDTH`
(480 px).

**`src/maestro/app/ui/cli_preview.CLIPreviewWindow`** is a `QMainWindow` that
reproduces the display behaviour of the full GUI without the configuration or telemetry
panels. It owns a `RuntimeController` and a `PreviewWidget`, connects signals
identically to `MainWindow`, and closes the Qt event loop when `Runtime.run()` returns.

**`src/maestro/app/main.py`** no longer imports or calls any OpenCV display function.
The Qt execution branch is imported lazily inside the `show_window` conditional so the
headless path retains zero Qt dependency at import time.

---

## Testing and Validation

- `tests/unit/test_ui_runtime_on_state.py` updated to use `show_window=False` in
  container fixtures, reflecting the decoupled display contract.
- `tests/unit/test_overlay_renderer.py` covers existing `OverlayRenderer` behaviour
  (landmark mirror policy, frame mutation guard, metadata preservation); the new
  `output_rgb` path follows the same structural contract.
- Full unit suite passes without suppressed tests; no cv2 display functions are
  exercised in the test environment.
- Black formatting applied; linting errors from the removed `QEvent` import resolved.

---

## Outcome

The inference runtime no longer holds any direct dependency on OpenCV's GUI subsystem.
Frame rendering is delegated to `RuntimeState` consumers through a single code path that
is shared between CLI preview mode and the full GUI dashboard. The Qt threading
violation introduced by calling `cv2.imshow()` from a `QThread` worker is eliminated.
`OverlayRenderer` centralises the BGR→RGB conversion, removing the redundant per-frame
channel reversal that previously occurred in the display layer. Both the CLI and GUI
entry points now use the same `PreviewWidget` component, ensuring visual consistency
across execution modes. No breaking changes were introduced to the existing headless
CLI path or to any public interface.
