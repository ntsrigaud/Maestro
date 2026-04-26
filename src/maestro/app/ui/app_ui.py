from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from maestro.app._paths import get_config_dir
from maestro.app.ui.main_window import MainWindow


def run_ui(config_path: str | None = None) -> int:
    """Bootstrap the QApplication and enter the event loop."""
    effective_path = config_path or str(get_config_dir())
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(config_path=effective_path)
    window.show()
    return app.exec()
