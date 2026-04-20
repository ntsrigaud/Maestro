# Work Log — Desktop Interaction Use Case Analysis and Implementation Roadmap

**Date:** 2026-04-19  
**Branch:** `docs/desktop-interaction-expansion`  
**Author:** @ntsrigaud

---

## Context

The gesture control system was originally framed as a presentation and meeting controller,
with the user standing ~1.5–3 m from the camera at a lectern. A systematic analysis
identified that geometry as out-of-distribution for the IPN Hand training data and
structurally unreliable for several feature blocks at distance. This branch documents that
finding and reframes the system around a desk-distance interaction model where the existing
production model (Macro-F1 = 0.96) is already adequate, then specifies a four-phase
implementation plan for extending the system to cover universal desktop scrolling, workspace
and tab navigation, view zoom, and OS overview mode.

---

## Deliverables

### `docs/development/desktop-interaction-analysis.md` (new, 137 lines)

**Section 1 — Why the presentation scenario is unreliable:**

Five root causes documented with evidence from the codebase:

| Root cause                                            | Key detail                                                                                      |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Dataset distribution mismatch                         | IPN recorded at 50–80 cm; presenter stands 150–300 cm — completely OOD                          |
| MediaPipe quality at distance                         | Hand < 80×80 px at 1280×720; landmark drop-out > 20% rejects windows via `min_valid_ratio: 0.8` |
| `fingertip_spread[63:68]` collapse                    | Inter-fingertip distances compress toward zero; `zoom_in` / `zoom_out` become indistinguishable |
| `joint_angles[134:144]` degradation                   | MCP/PIP angle estimation requires MediaPipe depth accuracy; degrades at small hand size         |
| `command_cooldown_ms: 1000` + `edge_trigger` mismatch | Window containing gesture peak is dropped by valid-ratio filter; edge trigger never fires       |

Environmental factors also documented: projector backwash, autofocus locking to the face, and low hand-background contrast.

**Section 2 — Refined use case scenarios:**

Four scenarios scoped to desk distance (≤ 100 cm), where the hand occupies ≥ 120×120 px and landmark quality is reliable:

| Scenario                   | Gestures                                                           | Implementation                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Universal Scroll           | `swipe_up` → `scroll_forward`, `swipe_down` → `scroll_back`        | `pyautogui.scroll()` at current cursor position — application-agnostic                                                                               |
| Workspace / Tab Navigation | `swipe_right` → `navigate_forward`, `swipe_left` → `navigate_back` | Per-platform workspace shortcuts or `Ctrl+Tab` / `Ctrl+Shift+Tab`; controlled by `navigation_mode` config field                                      |
| View Zoom                  | `zoom_in` → `zoom_in_view`, `zoom_out` → `zoom_out_view`           | Modifier+scroll injection: `Cmd+scroll` on macOS (not `Ctrl+scroll`, which activates the OS Accessibility magnifier), `Ctrl+scroll` on Windows/Linux |
| OS Overview Mode           | `open_palm_hold` → `mission_control`                               | Per-platform shortcut: `Ctrl+Up` / `Win+Tab` / `Super+A`                                                                                             |

Scroll direction convention: `swipe_up → scroll_forward` advances the document (natural/touchpad direction); command names describe document position, not gesture direction.

**Sections 3–4 — Design principles and command map:**

Five guiding constraints established: no pipeline changes, no retraining, single
`DesktopAdapter`, no repeated mappings, security preserved. Proposed
`command_map_desktop.yaml` specified with 8 logical command IDs; existing
`command_map_ipn.yaml` left untouched.

### `docs/development/desktop-interaction-roadmap.md` (new, 492 lines)

Four-phase implementation plan:

| Phase                     | Goal                                                      | Key changes                                                                                                                                                                                                                                             |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — Configuration         | Establish desktop command vocabulary and inference tuning | `command_map_desktop.yaml`; `configs/desktop/` profile directory; `AppConfig.command_map` + `AppConfig.navigation_mode` fields; bootstrap command map fix                                                                                               |
| 2 — Desktop Adapter       | `DesktopAdapter` covering all four scenarios              | `KeyboardAdapter.scroll()` and `KeyboardAdapter.zoom()` with clamping; `DesktopAdapter` with per-platform shortcut tables; platform normalisation + `ValueError` at construction for unknown platform; bootstrap `_build_dispatcher()` branch           |
| 3 — Navigation Tuning     | User-facing scroll and zoom sensitivity controls          | `DesktopConfig` schema block nested under `AppConfig`; `scroll_clicks` and `zoom_clicks` config fields; `linux_de` field for GNOME/KDE/other overview shortcut selection; `configs/desktop/app.yaml` and `configs/desktop/inference.yaml` profile files |
| 4 — Feedback Improvements | Surface active mode and pending command in the overlay    | (planned; not yet specified)                                                                                                                                                                                                                            |

Each phase specifies its test gate: all existing tests pass; named new tests for every new
schema field, adapter method, platform path, and error condition.

Security notes included inline:

- `zoom()` accepts a clamped `int`; `_ZOOM_MODIFIER` resolved from `sys.platform` at
  import time (not user-supplied)
- `try/finally` guarantees modifier key release if `scroll()` raises
- `"command"` added to `KeyboardAdapter._SAFE_KEY_NAMES` for macOS zoom

### Prettier Formatting Pass

Prettier applied across all touched Markdown and JSON experiment files (42 files total).
No semantic content changes — line wrapping, list indentation, and table spacing only.

---

## Outcome

The repository now has a precise, evidence-based analysis of the presentation scenario
limitation and a complete, phase-gated implementation plan for the desktop interaction
extension. No pipeline, model, or infrastructure code was modified. The plan is
self-contained: a developer can begin Phase 1 immediately from this document without
additional specification work.
