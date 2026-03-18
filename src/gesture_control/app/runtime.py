from __future__ import annotations

from gesture_control.app.bootstrap import AppContainer


class Runtime:
    def __init__(self, container: AppContainer) -> None:
        self._container = container

    def run(self) -> None:
        app_name = self._container.config.app.app_name
        print(f"{app_name} scaffold is ready. Runtime pipeline will be added in later phases.")
