from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import FramePacket, MultiHandPacket


class IHandLandmarkExtractor(Protocol):
    def extract(self, frame: FramePacket) -> MultiHandPacket: ...

    def close(self) -> None: ...
