from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import FramePacket


class IGazeIntentDetector(Protocol):
    def update(self, frame: FramePacket) -> bool: ...

    def close(self) -> None: ...
