from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import FramePacket, LandmarkPacket


class IHandLandmarkExtractor(Protocol):
    def extract(self, frame: FramePacket) -> LandmarkPacket: ...

    def close(self) -> None: ...
