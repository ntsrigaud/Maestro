# -*- mode: python ; coding: utf-8 -*-
#
# Maestro — PyInstaller build specification
#
# Usage:
#   pip install -e ".[ui,build]"
#   pyinstaller maestro.spec
#
# Output:
#   dist/Maestro.app          (macOS)
#   dist/Maestro/Maestro.exe  (Windows one-folder)
#
# NOTE: opencv-python (not headless) must be installed.  Install the [ui]
#       extra which replaces opencv-python-headless:
#         pip uninstall -y opencv-python-headless
#         pip install -e ".[ui,build]"
#
# Producing a truly cross-platform binary requires running this script on
# each target OS.  Use the GitHub Actions workflow (.github/workflows/build.yml)
# for automated cross-platform builds.

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(SPECPATH)  # noqa: F821 — injected by PyInstaller
SRC = ROOT / "src" / "maestro"
CONFIGS_SRC = ROOT / "configs"

# ---------------------------------------------------------------------------
# Data files bundled into the archive
# ---------------------------------------------------------------------------
datas = [
    # Maestro runtime configuration (user-editable defaults)
    (str(CONFIGS_SRC), "configs"),
    # Application icon (used by get_app_icon() at runtime)
    (str(ROOT / "docs" / "assets" / "icon.png"), "docs/assets"),
    # Offline bundled model
    (str(ROOT / "models"), "models"),
]

# MediaPipe ships its own model files and protobuf descriptors as data.
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
datas += mp_datas
datas += collect_data_files("google.protobuf")

# PySide6 plugins (platform, imageformats, etc.)
datas += collect_data_files("PySide6")

# ---------------------------------------------------------------------------
# Hidden imports that PyInstaller's static analysis misses
# ---------------------------------------------------------------------------
hiddenimports = mp_hidden + [
    # MediaPipe dynamic graph loading
    "mediapipe.python.solutions.hands",
    "mediapipe.python.solutions.face_mesh",
    # PyTorch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    # Hugging Face Hub
    "huggingface_hub",
    "huggingface_hub.file_download",
    # Maestro internals loaded via importlib / string references
    "maestro._defaults",
    "maestro.app._paths",
    "maestro.pipeline.hand_selection",
    "maestro.infrastructure.gaze",
    "maestro.infrastructure.dispatch.app_adapters.presentation_adapter",
    "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter",
]

# ---------------------------------------------------------------------------
# Binaries (shared libraries not captured automatically)
# ---------------------------------------------------------------------------
binaries = mp_binaries

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ROOT / "src" / "maestro" / "app" / "ui" / "app_ui.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude training / notebook extras to keep binary size down
        # NOTE: matplotlib must NOT be excluded — mediapipe.python.solutions.drawing_utils
        # imports matplotlib.pyplot at module level, which would cause a crash on startup.
        "sklearn",
        "seaborn",
        "pandas",
        "jupyter",
        "ipykernel",
        "notebook",
        # Test infrastructure
        "pytest",
        "pytest_qt",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

# ---------------------------------------------------------------------------
# Platform-specific output
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    # ── macOS: produce a .app bundle ────────────────────────────────────────
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="Maestro",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # GUI app — no terminal window
        icon="docs/assets/icon.icns",
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="Maestro",
    )
    app = BUNDLE(  # noqa: F821
        coll,
        name="Maestro.app",
        icon="docs/assets/icon.icns",
        bundle_identifier="com.ntsrigaud.maestro",
        info_plist={
            "CFBundleDisplayName": "Maestro",
            "CFBundleShortVersionString": "0.1.0",
            "NSCameraUsageDescription": (
                "Maestro requires camera access to detect hand gestures."
            ),
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )

else:
    # ── Windows / Linux: produce a one-folder executable ────────────────────
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="Maestro",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        icon="docs/assets/icon.ico",
    )
