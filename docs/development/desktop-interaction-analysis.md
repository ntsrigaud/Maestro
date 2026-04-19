# Desktop Interaction Use Case Analysis

**Date:** 2026-04-19  
**Branch:** `build/model-improvement`  
**Status:** Approved — drives `desktop-interaction-roadmap.md`

---

## 1. Why the Presentation Controller Scenario Is Unreliable

### 1.1 Dataset Distribution Mismatch

The IPN Hand dataset was recorded at close range: participants seated ~50–80 cm from a fixed webcam, hand occupying a large portion of the frame. The presentation controller scenario inverts this geometry — the presenter stands 1.5–3 m from the camera at a lectern. This is completely out-of-distribution for every feature block in feature-schema-v5, with no fine-tuning data bridging the gap.

### 1.2 MediaPipe Landmark Quality at Distance

MediaPipe Hands is optimised for near-field detection (~20–80 cm). At presentation distance (>1 m at 1280×720), the hand occupies fewer than 80×80 pixels.

| Effect                                | Consequence                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Landmark drop-out rate exceeds 20%    | `SlidingWindowSequenceBuffer` rejects windows below `min_valid_ratio: 0.8`; no inference fires                     |
| Sub-pixel joint position quantisation | Noise dominates `positions[0:63]`, `velocity[71:134]`, and `wrist_vel_raw[144:147]` relative to intentional motion |

### 1.3 Feature Representation Collapse

Two feature blocks are structurally unreliable at distance:

| Block              | Slice       | Problem                                                                                                            |
| ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `fingertip_spread` | `[63:68]`   | 5 inter-fingertip distances compress toward zero; `zoom_in` and `zoom_out` become indistinguishable                |
| `joint_angles`     | `[134:144]` | MCP/PIP angle estimation requires accurate MediaPipe depth, which degrades steeply when the hand is small in frame |

### 1.4 Configuration Mismatch

`command_cooldown_ms: 1000` with `edge_trigger_enabled: true` is tuned for rapid close-range gestures. At distance, the window containing the gesture peak is frequently dropped by the `min_valid_ratio` filter, so the edge trigger never fires and no command dispatches.

### 1.5 Environmental Factors

Presentation rooms expose the worst conditions for gesture control:

- **Lighting**: projector backwash and spotlights create high-contrast exposure mismatch that degrades MediaPipe's palm detector
- **Camera focus**: autofocus locked to the presenter's face defocuses the gesturing hand
- **Background contrast**: dark clothing against bright projected content reduces hand segmentation quality

---

## 2. Refined Use Case Scenarios

The system performs reliably when the user is seated within ~80–100 cm of the webcam — standard laptop or desktop monitor distance. This geometry matches the IPN recording setup exactly. At this range the hand occupies ≥ 120×120 pixels at 1280×720, keeping landmark drop-out below the `min_valid_ratio: 0.8` threshold.

`zoom_in`/`zoom_out` occupy a middle tier: `fingertip_spread[63:68]` and `joint_angles[134:144]` are less reliable than swipe velocity signals at this range, but the production threshold (0.85) and vote window (2) in `production_ipn.yaml` provide adequate filtering for occasional, deliberate pinch gestures. They are included in the desktop vocabulary for view-zoom scenarios where precision is more important than speed.

### 2.1 Universal Scroll Control

**Context**: User reads any scrollable content — web pages, PDFs, documentation, email, terminal output — while hands are otherwise engaged.

| Gesture      | Logical command  | Implementation                       |
| ------------ | ---------------- | ------------------------------------ |
| `swipe_up`   | `scroll_forward` | `pyautogui.scroll(-N)` — scroll down |
| `swipe_down` | `scroll_back`    | `pyautogui.scroll(+N)` — scroll up   |

`pyautogui.scroll()` injects a native OS scroll event at the current cursor position. It is **application-agnostic**: the same call advances the page in Chrome, Firefox, Preview, Acrobat, VS Code, or any other focused window. There is no need for per-application adapters. `wrist_vel_raw[144:147]` provides a clean vertical velocity signal at desk distance, making vertical swipes the top-1 most reliable gesture pair in production eval.

**Scroll direction convention**: `swipe_up → scroll_forward` scrolls content **downward** (advances the document); `swipe_down → scroll_back` scrolls content upward. This follows natural/touchpad scrolling — the content tracks the hand movement. It is the inverse of the legacy scroll-wheel convention and matches Apple Magic Trackpad and touchscreen behaviour. The command names (`scroll_forward` / `scroll_back`) describe document position, not gesture direction, keeping the semantics stable if the mapping is reconfigured.

### 2.2 Workspace and Tab Navigation

**Context**: User navigates between virtual desktops or browser tabs without a keyboard chord.

Horizontal swipes are discriminated primarily by `wrist_vel_raw[0]` (x-axis velocity). The OS shortcut depends on the configured `navigation_mode`:

| Gesture       | Logical command    | `navigation_mode: workspace`                           | `navigation_mode: tabs` |
| ------------- | ------------------ | ------------------------------------------------------ | ----------------------- |
| `swipe_right` | `navigate_forward` | macOS `Ctrl+→` / Win `Ctrl+Win+→` / Linux `Ctrl+Alt+→` | `Ctrl+Tab`              |
| `swipe_left`  | `navigate_back`    | macOS `Ctrl+←` / Win `Ctrl+Win+←` / Linux `Ctrl+Alt+←` | `Ctrl+Shift+Tab`        |

The same logical commands (`navigate_forward`, `navigate_back`) serve both workspace and tab switching; the target is a single config field (`navigation_mode`), not a per-application adapter. The tab shortcuts are identical across Chrome, Firefox, Safari, and Edge.

### 2.3 View Zoom

**Context**: User reads a dense PDF, web page, or document and needs to magnify a region without reaching for the keyboard or trackpad.

| Gesture    | Logical command | Implementation                                  |
| ---------- | --------------- | ----------------------------------------------- |
| `zoom_in`  | `zoom_in_view`  | `Ctrl` held + `pyautogui.scroll(+N)` — zoom in  |
| `zoom_out` | `zoom_out_view` | `Ctrl` held + `pyautogui.scroll(-N)` — zoom out |

Holding `Ctrl` while injecting a scroll event is the universal zoom mechanism on all major desktop OSes. It works identically in Chrome, Firefox, Safari, Edge, PDF viewers (Preview, Acrobat, Evince, Okular), image viewers, VS Code, and Office applications — no per-application mapping is required. This mirrors how a physical scroll wheel with `Ctrl` held triggers zoom.

**Reliability note**: `zoom_in`/`zoom_out` have lower recognition stability than swipe gestures at desk distance due to `fingertip_spread[63:68]` compression. They are suitable for occasional, deliberate use — not rapid-fire repeated triggers. The production threshold (0.85) and vote window (2) suppress most false positives.

### 2.4 OS Overview Mode (Mission Control)

**Context**: User wants to switch tasks by invoking the OS-level window/space overview without reaching for the keyboard or trackpad.

| Gesture          | Logical command   | macOS            | Windows   | Linux (GNOME) | Linux (KDE) |
| ---------------- | ----------------- | ---------------- | --------- | ------------- | ----------- |
| `open_palm_hold` | `mission_control` | `Ctrl+Up` / `F3` | `Win+Tab` | `Super+A`     | `Super+W`   |

`open_palm_hold` is well-suited for this role: it has the highest vote window (`vote_window: 3`) and confidence threshold (`0.90`) in `production_ipn.yaml`, making it the most deliberate gesture in the vocabulary — resistant to accidental triggering. The `edge_trigger_enabled: true` setting ensures the overview fires once per hold event rather than repeatedly. Dismissal is handled by the OS (pressing the gesture again or the Escape key); the system does not need to track the overview's open/closed state.

---

## 3. Core Design Principles for the Extension

1. **No pipeline changes.** Feature engineering, LSTM model, stabiliser, and gaze activation are untouched.
2. **No retraining required.** The existing production model (Macro-F1 = 0.96) is adequate for all four scenarios at desk distance.
3. **Single unified adapter.** One `DesktopAdapter` covers all desktop scenarios. Per-application adapters are not needed because scroll and zoom are application-agnostic OS events; navigation and overview use a config flag rather than per-app shortcut tables.
4. **No repeated mappings.** Commands that behave identically across all applications (scroll, Ctrl+scroll zoom) are implemented once via OS event injection; only genuine per-OS variants (workspace shortcuts, overview key) appear in platform tables.
5. **Full gesture vocabulary active.** All 8 model classes are mapped to a logical command. `zoom_in`/`zoom_out` are activated for view-zoom via Ctrl+scroll injection rather than excluded.

---

## 4. Gesture–Command Mapping Redesign

Current `command_map_ipn.yaml` uses presentation-domain identifiers. The desktop expansion requires application-agnostic logical commands. `open_palm_hold` is promoted to `mission_control`; `zoom_in`/`zoom_out` are promoted to `zoom_in_view`/`zoom_out_view` via Ctrl+scroll injection.

**Proposed `command_map_desktop.yaml`:**

```yaml
version: v2
gesture_commands:
  swipe_up: scroll_forward # universal: advance page in any focused app
  swipe_down: scroll_back # universal: go back in any focused app
  swipe_right: navigate_forward # next workspace or next tab (navigation_mode)
  swipe_left: navigate_back # previous workspace or previous tab
  zoom_in: zoom_in_view # Ctrl+scroll up — universal view zoom in
  zoom_out: zoom_out_view # Ctrl+scroll down — universal view zoom out
  open_palm_hold: mission_control # OS overview mode (Mission Control / Task View / Activities)
  unknown: no_action
```

The existing `command_map_ipn.yaml` is preserved. The active map is selected by a `command_map` key in `app.yaml`.

---

**See also:** [`desktop-interaction-roadmap.md`](desktop-interaction-roadmap.md) for the phase-by-phase implementation plan.
