from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gesture_control.app.ui.main_window import MainWindow

_DEFAULT_CONFIG_PATH = "configs"


def run_ui(config_path: str = _DEFAULT_CONFIG_PATH) -> int:
    """Bootstrap the QApplication and enter the event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(config_path=config_path)
    window.show()
    return app.exec()
