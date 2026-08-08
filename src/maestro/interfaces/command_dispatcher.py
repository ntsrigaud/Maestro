from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import DispatchResult, ExecutableCommand


class ICommandDispatcher(Protocol):
    def dispatch(self, command: ExecutableCommand) -> DispatchResult: ...
    def is_target_app_focused(self) -> bool: ...
