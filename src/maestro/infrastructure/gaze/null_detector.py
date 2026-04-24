"""No-op gaze detector used when ``gaze.enabled = false``."""

from __future__ import annotations

from maestro.domain.entities import FramePacket


class NullGazeDetector:
    """Always-permitted gaze detector (null-object pattern).

    Implements ``IGazeIntentDetector`` structurally (duck-typed Protocol).

    Substituted for ``HeadPoseGazeDetector`` when ``gaze.enabled = false``
    in ``configs/inference.yaml``.  Restores the pre-gaze pipeline behaviour
    without any conditional branches in the runtime loop or bootstrap.
    """

    def update(self, frame: FramePacket) -> bool:  # noqa: ARG002
        """Return ``True`` unconditionally; gaze gate is always open."""
        return True

    def close(self) -> None:
        """No-op; there are no resources to release."""
