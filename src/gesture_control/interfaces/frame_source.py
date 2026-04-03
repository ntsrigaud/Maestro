from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import FramePacket


class IFrameSource(Protocol):
    def read(self) -> FramePacket: ...

    def close(self) -> None: ...
