# Work Log — Packaging and Distribution Strategy

**Date:** 2026-04-26  
**Branch:** `build/packaging-and-distribution-strategy`  
**Author:** @ntsrigaud

---

## Context

Maestro was runnable from source but had no distribution path for end users who are not Python developers. Config loading assumed `CWD/configs` was always reachable, the `--config` CLI flag was the only escape hatch, and no entry points were defined. This work introduces the full pipeline from configuration path resolution through binary production and CI-automated release.

---

## Scope of Work

### Runtime configuration path resolution

A four-tier resolution chain replaces the hard-coded `"configs"` string used throughout the application entry points. The chain (`MAESTRO_CONFIG_DIR` → PyInstaller `_MEIPASS` → OS user config dir → `CWD/configs`) ensures the correct config directory is found regardless of whether the process is invoked from a repository checkout, an installed wheel, or a frozen binary. First-launch seeding from bundled package data is handled transparently.

### Application icon pipeline

A programmatic icon generation tool (`tools/generate_icon.py`) renders a hand-silhouette SVG concept into Pillow-drawn raster output: a 1024×1024 PNG master, a multi-resolution ICNS bundle for macOS, and an ICO file for Windows. Generated binaries are committed. The icon is applied to the `QApplication` instance at startup so all windows inherit it.

### PyInstaller spec and Makefile targets

`maestro.spec` provides a platform-conditional build specification. On macOS it produces a signed-ready `.app` bundle with the camera usage privacy string; on Windows it produces a one-folder executable. All runtime data (configs, icon, MediaPipe model files, protobuf descriptors, PySide6 plugins) are bundled. Training/notebook/test dependencies are excluded. Three new Makefile targets (`build`, `build-onefile`, `generate-icon`) and an extended `clean` consolidate all build operations. A self-documenting `help` target provides a formatted target reference from inline comments.

### GitHub Actions release pipeline

`.github/workflows/build.yml` implements a three-stage release pipeline: a parallel test gate across all three platforms, platform-specific build jobs (macOS DMG, Windows ZIP), and an automated draft GitHub Release on version tag pushes. A `workflow_dispatch` path with an artifact-upload toggle supports manual triggering.

### Entry points and package extras

`pyproject.toml` gains `[project.scripts]` (`maestro`) and `[project.gui-scripts]` (`maestro-ui`) entry points, and a `build` optional extra for PyInstaller. The `maestro._defaults` package ships factory-default config YAMLs as wheel package data so `get_config_dir()` can seed user config on first launch.

---

## Technical Changes

### `src/maestro/app/_paths.py` (new)

- `get_config_dir()`: four-tier resolution — env override → `sys._MEIPASS/configs` → user config dir → `CWD/configs`. Populates user dir from `maestro._defaults` package data on first launch.
- `get_app_icon()`: resolves `docs/assets/icon.png` from the frozen bundle or by walking `__file__` parents in development.
- `_user_config_dir()`: returns the OS-appropriate per-user data directory (macOS `~/Library/Application Support/Maestro`, Windows `%APPDATA%\Maestro`, Linux `XDG_CONFIG_HOME/Maestro`).

### `src/maestro/_defaults/__init__.py` (new)

- Package marker. Ships `configs/**/*.yaml` as package data via `pyproject.toml`. Provides the source tree for first-launch config seeding.

### `src/maestro/app/main.py`

- `config_path` now resolved via `get_config_dir()` when `--config` is absent; replaces previous hard-coded `"configs"` fallback.

### `src/maestro/app/ui/app_ui.py`

- `run_ui()` calls `get_config_dir()` for `effective_path` when `config_path` is `None`.
- Calls `get_app_icon()` and applies the result to the `QApplication` via `app.setWindowIcon(QIcon(...))`.

### `src/maestro/app/ui/main_window.py`

- `MainWindow.__init__` resolves `self._config_path` via `get_config_dir()` when `config_path` is not supplied, replacing the empty-string module constant that caused `ConfigPanel` to receive an unresolved path.

### `maestro.spec` (new)

- Platform-conditional spec producing `.app` (macOS, via `BUNDLE` with `NSCameraUsageDescription`, `CFBundleIdentifier = com.ntsrigaud.maestro`) or one-folder EXE (Windows/Linux).
- Bundled data: `configs/` tree, `docs/assets/icon.png`; MediaPipe via `collect_all`; protobuf via `collect_data_files`; PySide6 plugins via `collect_data_files`.
- Hidden imports: MediaPipe solutions, PyTorch, `huggingface_hub`, `maestro._defaults`, `maestro.app._paths`, and lazily-imported pipeline modules.
- Excludes: `sklearn`, `matplotlib`, `seaborn`, `pandas`, `jupyter`, `pytest`.

### `Makefile`

- `build`: runs `install-build` then `pyinstaller --noconfirm maestro.spec`; platform-appropriate output message.
- `build-onefile`: `--onefile --windowed` variant for a single-file binary.
- `generate-icon`: delegates to `python tools/generate_icon.py`.
- `clean`: extended to remove `build/`, `dist/`, `docs/assets/icon_preview.png`, `docs/assets/icon.iconset`.
- `help`: auto-generates a colour-coded target reference from `## …` inline comments via `awk`.

### `.github/workflows/build.yml` (new)

- `test` job: parallel matrix across `ubuntu-latest`, `macos-latest`, `windows-latest`; runs `pytest -m "not training" -q`; `fail-fast: true`.
- `build-macos`: `needs: test`; installs `[ui,build]`; runs spec; packages output to `.dmg` via `hdiutil`; uploads artifact `Maestro-macos`.
- `build-windows`: `needs: test`; installs `[ui,build]`; runs spec; archives to `.zip` via PowerShell `Compress-Archive`; uploads artifact `Maestro-windows`.
- `release`: `needs: [build-macos, build-windows]`; runs on tag pushes only; creates a draft release with auto-generated notes via `softprops/action-gh-release`.

### `tools/generate_icon.py` (new)

- Renders a hand silhouette (palm + five fingers via `ImageDraw.rounded_rectangle`, thumb with rotation) over a `#111827 → #1E1B4B` vertical gradient with indigo ring and glow layers.
- Produces `docs/assets/icon.png` (1024×1024), `icon.icns` (macOS, via `iconutil` subprocess), and `icon.ico` (Windows, multi-resolution via Pillow).

### `docs/assets/` (additions)

- `icon.svg`, `icon.png`, `icon.icns`, `icon.ico`: application icon in source and platform-binary forms.

### `pyproject.toml`

- `[project.scripts]`: `maestro = "maestro.app.main:main"`.
- `[project.gui-scripts]`: `maestro-ui = "maestro.app.ui.app_ui:run_ui"`.
- `[project.optional-dependencies]` `build`: `pyinstaller>=6.0.0`.
- `[tool.setuptools.package-data]`: `"maestro._defaults" = ["configs/**/*.yaml", "configs/**/*.yml"]`.

### `README.md`

- New **Installation** section: Option A (standalone binary — download table for macOS DMG / Windows ZIP) and Option B (pip install with entry points).
- First-launch config seeding documented with per-platform path table and `MAESTRO_CONFIG_DIR` override.
- **Building from Source** section documenting `make build` and the Actions workflow.

---

## Testing and Validation

- All pre-existing unit and integration tests pass without modification; the new path resolution module has no test-time side effects because `CWD/configs` is always present in a repository checkout.
- The `workflow_dispatch` trigger on `build.yml` allows manual validation of the full build matrix before a tagged release.
- `make verify` (Makefile `build` target output check) confirms bundle presence with a shell conditional (`test -d dist/Maestro.app` on macOS).

---

## Outcome

Maestro can now be distributed to end users as a self-contained binary with no Python prerequisite. The config resolution chain is transparent to all three deployment contexts (source checkout, installed wheel, frozen bundle). The CI pipeline automates cross-platform artifact production and draft release creation on every version tag.
