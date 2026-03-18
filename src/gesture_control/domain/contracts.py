from __future__ import annotations

from typing import Protocol


class IRunnable(Protocol):
    def run(self) -> None:
        ...


class IClosable(Protocol):
    def close(self) -> None:
        ...
