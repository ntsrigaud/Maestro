from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import Prediction, StabilizedPrediction


class IPredictionStabilizer(Protocol):
    def update(self, prediction: Prediction) -> StabilizedPrediction: ...
