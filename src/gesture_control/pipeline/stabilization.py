from __future__ import annotations

from collections import deque

from gesture_control.domain.entities import Prediction, StabilizedPrediction


class PredictionStabilizer:
    """Stabilizes online predictions via thresholding + EMA + temporal voting."""

    def __init__(
        self,
        *,
        default_threshold: float,
        class_thresholds: dict[str, float],
        vote_window: int = 3,
        class_vote_windows: dict[str, int] | None = None,
        ema_alpha: float = 0.4,
        unknown_label: str = "unknown",
    ) -> None:
        if vote_window <= 0:
            raise ValueError("vote_window must be > 0")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0.0, 1.0]")
        if class_vote_windows is not None:
            for cls, w in class_vote_windows.items():
                if w <= 0:
                    raise ValueError(f"class_vote_windows[{cls!r}] must be > 0")

        self._default_threshold = default_threshold
        self._class_thresholds = dict(class_thresholds)
        self._vote_window = vote_window
        self._class_vote_windows: dict[str, int] = dict(class_vote_windows) if class_vote_windows else {}
        self._ema_alpha = ema_alpha
        self._unknown_label = unknown_label

        self._ema_probs: dict[str, float] = {}
        max_window = max([vote_window] + list(self._class_vote_windows.values()))
        self._history: deque[str] = deque(maxlen=max_window)

    def update(self, prediction: Prediction) -> StabilizedPrediction:
        smoothed_probs = self._smooth_probs(prediction.class_probs)
        label, confidence = self._argmax(smoothed_probs)

        threshold = self._class_thresholds.get(label, self._default_threshold)
        if confidence < threshold:
            candidate = self._unknown_label
        else:
            candidate = label

        self._history.append(candidate)

        if candidate == self._unknown_label:
            return StabilizedPrediction(
                label=self._unknown_label,
                confidence=float(confidence),
                accepted=False,
                reason="below_threshold_or_unknown",
            )

        if self._is_temporally_consistent(candidate):
            return StabilizedPrediction(
                label=candidate,
                confidence=float(confidence),
                accepted=True,
                reason=None,
            )

        return StabilizedPrediction(
            label=candidate,
            confidence=float(confidence),
            accepted=False,
            reason="temporal_inconsistency",
        )

    def _is_temporally_consistent(self, candidate: str) -> bool:
        window = self._class_vote_windows.get(candidate, self._vote_window)
        if len(self._history) < window:
            return False
        return all(label == candidate for label in list(self._history)[-window:])

    def _smooth_probs(self, probs: dict[str, float]) -> dict[str, float]:
        if not self._ema_probs:
            self._ema_probs = dict(probs)
            return dict(self._ema_probs)

        all_labels = set(self._ema_probs) | set(probs)
        next_probs: dict[str, float] = {}
        for label in all_labels:
            prev = self._ema_probs.get(label, 0.0)
            curr = probs.get(label, 0.0)
            next_probs[label] = self._ema_alpha * curr + (1.0 - self._ema_alpha) * prev

        self._ema_probs = next_probs
        return dict(next_probs)

    @staticmethod
    def _argmax(probs: dict[str, float]) -> tuple[str, float]:
        if not probs:
            return "unknown", 0.0
        label, confidence = max(probs.items(), key=lambda item: item[1])
        return str(label), float(confidence)
