# Maestro

Production-grade webcam-based gesture recognition system for hands-free presentation control, powered by MediaPipe Hands and a two-stream LSTM classifier.

## System Overview

The system recognizes a 10-class runtime vocabulary from a live webcam feed (9 active gestures + `unknown`) and executes validated presentation controls in real time on consumer laptops without a GPU.

**Core pipeline:**

```
Camera → MediaPipe Hands → Hand Selection → Feature Engineering → Sliding Window Buffer
→ LSTM Inference → Prediction Stabilization → Activation Controller
→ Command Dispatch → Application Adapter → OS Keyboard/Pointer Events
```

A **PySide6 graphical interface** provides runtime control, live telemetry monitoring, configuration editing, and model selection — all without interfering with the inference pipeline. The inference loop runs in a background `QThread`; the UI receives state updates through Qt signal-slot connections at approximately 10 Hz.

For developers who prefer script-based invocation, `scripts/run_ui.py` is available as a thin wrapper.

**Gesture vocabulary** (hybrid LD-ConGR+Jester+IPN, 10 classes):

| Gesture                 | Command                   | Source   |
| ----------------------- | ------------------------- | -------- |
| `fist`                  | Erase annotations         | LD-ConGR |
| `swiping_right`         | Next slide                | LD-ConGR |
| `swiping_left`          | Previous slide            | LD-ConGR |
| `swiping_up`            | Start presentation        | LD-ConGR |
| `swiping_down`          | Stop presentation         | LD-ConGR |
| `zooming_in_full_hand`  | Zoom in view              | Jester   |
| `zooming_out_full_hand` | Zoom out view             | Jester   |
| `point_one`             | Continuous laser pointer  | IPN      |
| `point_two`             | Continuous annotation pen | IPN      |
| `unknown`               | No action (noise class)   | IPN      |

Discrete commands are routed through `GestureActivationController` + `CommandDispatcher`.
`point_one` and `point_two` are continuous gestures handled directly by runtime trackers.

## Installation

Download the pre-built installer for your platform from the [GitHub Releases](https://github.com/ntsrigaud/Maestro/releases) page. The application is completely standalone and does not require Python or internet access to run (the AI models are bundled directly into the app).

### macOS
1. Download `Maestro-macos.dmg` and double-click to mount it.
2. Drag `Maestro.app` into your **Applications** folder.
3. **Security Bypass (Gatekeeper)**: Because this application is not signed with a paid Apple Developer ID, macOS will show a "cannot be opened" or "malware" warning. To bypass this:
   - **Method A (GUI):** Open **System Settings > Privacy & Security**, scroll down to the Security section, and click **Open Anyway** next to Maestro.
   - **Method B (Terminal):** Open `Terminal` and run the following command to remove the quarantine flag:
     ```bash
     xattr -cr /Applications/Maestro.app
     ```
4. Launch Maestro from your Applications folder.
5. **Accessibility Permissions**: Maestro requires Accessibility permissions to inject keystrokes into PowerPoint. When first executing a gesture, macOS may prompt you. If not, open **System Settings > Privacy & Security > Accessibility** and manually enable Maestro.

### Windows
1. Download `Maestro_Setup.exe`.
2. Double-click the setup file to install Maestro.
3. **Important**: Because this application is not signed with an Authenticode certificate, Windows SmartScreen may show a "Windows protected your PC" warning. Click **More info** and then click **Run anyway**.
4. Launch Maestro from your Start Menu.

On first launch the application copies its factory-default configuration to:

| Platform | Location                                         |
| -------- | ------------------------------------------------ |
| macOS    | `~/Library/Application Support/Maestro/configs/` |
| Windows  | `%APPDATA%\Maestro\configs\`                     |
| Linux    | `~/.config/Maestro/configs/`                     |

Edit the YAML files there to change camera, hand preference, or gesture mappings. Set `MAESTRO_CONFIG_DIR` to override the location entirely.

## Project Architecture

See `docs/architecture/` for the full system specification, module contracts, and design rationale.

Key directories:

| Directory      | Purpose                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/maestro/` | Core package (domain, interfaces, infrastructure, pipeline, training, app/ui)                                                      |
| `configs/`     | YAML configuration (app, inference, model, gestures, thresholds)                                                                   |
| `scripts/`     | UI entry point (`run_ui.py`), data generation, and training utilities                                                                  |
| `models/`      | Local training artifacts (`.gitignore`d); production model hosted on [Hugging Face](https://huggingface.co/ntsrigaud/maestro-lstm-hybrid) |
| `tests/`       | Unit, integration, and system test suites                                                                                          |
| `docs/`        | Architecture docs, guides, and decision records                                                                                    |

### GUI Architecture

The PySide6 UI (`src/maestro/app/ui/`) is a non-blocking control surface:

- `RuntimeController` owns a `QThread` worker that runs `build_container()` and `Runtime.run()`.
- `Runtime` receives an optional `on_state` callback; the worker thread throttles callback emissions to ~10 Hz and forwards them as Qt signals to the main thread.
- `MainWindow` mediates all signal routing between panels and the controller; no panel communicates with the controller directly.
- Configuration changes are validated in-memory by Pydantic; YAML files are written only after the runtime stops and only on explicit user action.
- The main UI window is the cross-platform control surface, and a `QSystemTrayIcon` supplements it with ambient gesture feedback (colour-coded: grey / green / amber) when available.

The complete development history is recorded in [docs/architecture/](docs/architecture/) and [docs/archive/work-logs/](docs/archive/work-logs/).

## License

Maestro's source code is released under the [Apache License 2.0](LICENSE). Third-party
components and their licenses are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

The pretrained model weights are licensed separately, for **academic and non-commercial
research use only** (see [MODEL_LICENSE.md](MODEL_LICENSE.md)), because they were trained on
LD-ConGR, Jester and IPN Hand, whose terms restrict commercial use. The installers bundle these
weights and are subject to the same restriction.

## Citation

If you use Maestro in academic work, please cite it using [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button). A reference to the accompanying paper will be added
after publication.
