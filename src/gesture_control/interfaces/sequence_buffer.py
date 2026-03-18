from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import FeatureFrame, SequenceWindow


class ISequenceBuffer(Protocol):
    def push(self, feature: FeatureFrame) -> None: ...

    def ready(self) -> bool: ...

    def window(self) -> SequenceWindow: ...
