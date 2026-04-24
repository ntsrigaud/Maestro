from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import ExecutableCommand, StabilizedPrediction


class IActivationController(Protocol):
    def route(self, prediction: StabilizedPrediction) -> ExecutableCommand | None: ...
