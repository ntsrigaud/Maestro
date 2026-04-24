from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import FramePacket, RuntimeState


class IFeedbackRenderer(Protocol):
    def render(self, frame: FramePacket, state: RuntimeState) -> FramePacket: ...
