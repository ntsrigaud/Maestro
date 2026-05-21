from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from maestro.app._paths import get_app_icon, get_config_dir
from maestro.app.ui.main_window import MainWindow


def run_ui(config_path: str | None = None) -> int:
    """Bootstrap the QApplication and enter the event loop."""
    effective_path = config_path or str(get_config_dir())
    app: QApplication = QApplication.instance() or QApplication(sys.argv)  # type: ignore[assignment]
    app.setQuitOnLastWindowClosed(False)
    icon_path = get_app_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow(config_path=effective_path)  # noqa: F841  kept to prevent GC
    # Auto-activate after the event loop starts so the _RuntimeThread is
    # spawned while app.exec() is running, not during widget construction.
    QTimer.singleShot(0, window.auto_start)
    return app.exec()
