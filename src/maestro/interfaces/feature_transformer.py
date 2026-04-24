from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import FeatureFrame, LandmarkPacket


class IFeatureTransformer(Protocol):
    def transform(self, landmarks: LandmarkPacket) -> FeatureFrame: ...
