from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import DispatchResult, ExecutableCommand


class ICommandDispatcher(Protocol):
    def dispatch(self, command: ExecutableCommand) -> DispatchResult: ...
