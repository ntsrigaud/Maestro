from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import DispatchResult


class IApplicationAdapter(Protocol):
    """Contract for platform-specific command execution adapters.

    Implementations translate logical command IDs (e.g. ``"next_slide"``) into
    concrete OS-level actions.  Each adapter targets one application and
    exposes its name via ``target_app`` so that dispatchers can log and route
    correctly.
    """

    @property
    def target_app(self) -> str:
        """Identifier of the target application (e.g. ``"powerpoint"``)."""
        ...

    def execute(self, command_id: str) -> DispatchResult:
        """Execute the action corresponding to *command_id*.

        Returns a :class:`DispatchResult` describing success or failure without
        raising exceptions; implementations catch and log adapter-level errors
        internally.
        """
        ...

    def is_target_app_focused(self) -> bool:
        """Return True if the target application is currently in the foreground."""
        ...
