from __future__ import annotations

import logging
import time
from collections import deque

from maestro.domain.entities import DispatchResult, ExecutableCommand
from maestro.infrastructure.dispatch.app_adapters.base import (
    IApplicationAdapter,
)

LOGGER = logging.getLogger(__name__)


class CommandDispatcher:
    """Secure command dispatcher that routes commands to application adapters.

    Responsibilities
    ----------------
    - **Allowlist enforcement**: Only command IDs explicitly listed in
      ``allowed_commands`` are passed to adapters.  Any unlisted command ID
      is rejected immediately with a logged warning and a failed
      :class:`DispatchResult`; it never reaches the OS.
    - **Per-command rate limiting**: A deque-based sliding-window rate limiter
      ensures that at most ``rate_limit_per_window`` commands are dispatched
      within any ``rate_limit_window_ms`` interval.  Commands that would exceed
      the limit are dropped (not queued).
    - **Structured logging**: Every dispatch attempt (allowed / rejected /
      rate-limited) is recorded with ``command_id``, ``source_gesture``, and
      outcome so that false-trigger debugging is straightforward.
    - **Adapter isolation**: Individual adapter exceptions are caught here and
      returned as a failed :class:`DispatchResult` so the runtime loop is never
      interrupted by adapter failures.

    Parameters
    ----------
    adapter:
        The application adapter to invoke for successful commands.
    allowed_commands:
        Strict allowlist of command_id strings the dispatcher will forward.
        Any command_id absent from this set is silently rejected.
    rate_limit_per_window:
        Maximum number of commands that may be dispatched in any rolling
        ``rate_limit_window_ms`` millisecond window.  Set to 0 to disable.
    rate_limit_window_ms:
        Duration of the sliding-window rate-limit bucket in milliseconds.
    """

    def __init__(
        self,
        *,
        adapter: IApplicationAdapter,
        allowed_commands: set[str] | frozenset[str],
        rate_limit_per_window: int = 5,
        rate_limit_window_ms: int = 1000,
    ) -> None:
        if rate_limit_per_window < 0:
            raise ValueError("rate_limit_per_window must be >= 0")
        if rate_limit_window_ms <= 0:
            raise ValueError("rate_limit_window_ms must be > 0")

        self._adapter = adapter
        self._allowed_commands: frozenset[str] = frozenset(allowed_commands)
        self._rate_limit: int = rate_limit_per_window
        self._window_ms: int = rate_limit_window_ms
        # Timestamps (ms) of recent successful dispatches used for rate limiting.
        self._dispatch_timestamps: deque[int] = deque()

    # ------------------------------------------------------------------
    # Public interface (satisfies ICommandDispatcher protocol)
    # ------------------------------------------------------------------

    def dispatch(self, command: ExecutableCommand) -> DispatchResult:
        """Validate and execute *command* through the configured adapter.

        The following guards are applied in order:

        1. Command ID allowlist check — unknown IDs are rejected immediately.
        2. Rate limit check — excess commands within the window are dropped.
        3. Adapter execution — the result is returned to the caller.
        """
        # ---- Guard 1: Allowlist ----
        if command.command_id not in self._allowed_commands:
            detail = f"rejected_by_allowlist: command_id={command.command_id}"
            LOGGER.warning(
                "DISPATCH_REJECTED allowlist command_id=%s gesture=%s",
                command.command_id,
                command.source_gesture,
            )
            return DispatchResult(
                success=False,
                command_id=command.command_id,
                target_app=self._adapter.target_app,
                detail=detail,
            )

        # ---- Guard 2: Rate limit ----
        now_ms = self._now_ms()
        if self._rate_limit > 0:
            self._prune_old_timestamps(now_ms)
            if len(self._dispatch_timestamps) >= self._rate_limit:
                detail = (
                    f"rate_limited: {len(self._dispatch_timestamps)} dispatches "
                    f"within last {self._window_ms} ms"
                )
                LOGGER.warning(
                    "DISPATCH_RATE_LIMITED command_id=%s gesture=%s",
                    command.command_id,
                    command.source_gesture,
                )
                return DispatchResult(
                    success=False,
                    command_id=command.command_id,
                    target_app=self._adapter.target_app,
                    detail=detail,
                )

        # ---- Guard 3: Adapter execution ----
        LOGGER.info(
            "DISPATCH_START command_id=%s gesture=%s target_app=%s",
            command.command_id,
            command.source_gesture,
            self._adapter.target_app,
        )
        try:
            result = self._adapter.execute(command.command_id)
        except Exception as exc:
            LOGGER.exception(
                "DISPATCH_ADAPTER_ERROR command_id=%s error=%s",
                command.command_id,
                exc,
            )
            return DispatchResult(
                success=False,
                command_id=command.command_id,
                target_app=self._adapter.target_app,
                detail=f"adapter_exception: {exc}",
            )

        if result.success:
            self._dispatch_timestamps.append(now_ms)
            LOGGER.info(
                "DISPATCH_OK command_id=%s target_app=%s detail=%s",
                result.command_id,
                result.target_app,
                result.detail,
            )
        else:
            LOGGER.warning(
                "DISPATCH_FAILED command_id=%s target_app=%s detail=%s",
                result.command_id,
                result.target_app,
                result.detail,
            )

        return result

    def is_target_app_focused(self) -> bool:
        """Return True if the configured application adapter's target app is focused."""
        if hasattr(self._adapter, "is_target_app_focused"):
            return bool(self._adapter.is_target_app_focused())
        return True

    # ------------------------------------------------------------------
    # Properties for introspection
    # ------------------------------------------------------------------

    @property
    def allowed_commands(self) -> frozenset[str]:
        return self._allowed_commands

    @property
    def adapter(self) -> IApplicationAdapter:
        return self._adapter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_timestamps(self, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms
        while self._dispatch_timestamps and self._dispatch_timestamps[0] <= cutoff:
            self._dispatch_timestamps.popleft()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CommandDispatcher(app={self._adapter.target_app!r}, "
            f"allowed={sorted(self._allowed_commands)}, "
            f"rate_limit={self._rate_limit}/{self._window_ms}ms)"
        )
