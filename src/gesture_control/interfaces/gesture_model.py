from __future__ import annotations

from typing import Protocol

from gesture_control.domain.entities import Prediction, SequenceWindow


class IGestureModel(Protocol):
    def predict(self, sequence: SequenceWindow) -> Prediction:
        ...
