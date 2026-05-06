# DEPRECATED: retired in STGCN migration. This interface is no longer part of
# the inference hot path. Scheduled for deletion after Phase 5 validation.

from __future__ import annotations

from typing import Protocol

from maestro.domain.entities import FeatureFrame, LandmarkPacket


class IFeatureTransformer(Protocol):
    def transform(self, landmarks: LandmarkPacket) -> FeatureFrame: ...
