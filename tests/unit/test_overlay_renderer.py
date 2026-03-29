from __future__ import annotations

import numpy as np
import pytest

from gesture_control.domain.entities import FramePacket, RuntimeState
from gesture_control.infrastructure.feedback.overlay_renderer import OverlayRenderer


def _make_frame(width: int = 100, height: int = 100) -> FramePacket:
    """Create a plain black BGR frame for testing."""
    return FramePacket(
        frame=np.zeros((height, width, 3), dtype=np.uint8),
        timestamp_ms=1000,
        frame_id=0,
    )


def _make_state(**info_overrides: object) -> RuntimeState:
    return RuntimeState(
        mode="idle",
        last_label="unknown",
        last_confidence=0.0,
        info={"landmarks": [], **info_overrides},
    )


# ---------------------------------------------------------------------------
# Coordinate orientation policy: landmark x positions on the rendered frame
# ---------------------------------------------------------------------------

class TestLandmarkMirrorPolicy:
    """
    Coordinate orientation policy:

    The ML pipeline always works in the original (non-mirrored) camera
    coordinate space where x=0 is the camera-left edge and x=1 is the
    camera-right edge.  When mirror_display=True the canvas is flipped
    horizontally so the display looks like a selfie mirror.  Landmark
    dots must land on the correct side of the flipped canvas, which
    requires inverting the x coordinate:

        display_x_pixel = (width - 1) - raw_x_pixel  ≈  (1 - raw_x) * width

    The landmark dot is drawn with color (B=0, G=255, R=0) (pure green).
    Text overlays use color (B=0, G=255, R=255) (yellow).  Checking that
    the Red channel is 0 at a pixel distinguishes the dot from any text
    that overlaps in the Green channel.
    """

    # Use a frame large enough that the landmark at y=0.85 (row ≈ 408 in 480px
    # height) is well below all text lines (which end before row ~140).
    WIDTH = 640
    HEIGHT = 480

    def _landmark_at(self, raw_x: float) -> list[list[float]]:
        return [[raw_x, 0.85, 0.0]]

    def _dot_col_raw(self, raw_x: float) -> int:
        return int(raw_x * self.WIDTH)

    def _dot_col_mirror(self, raw_x: float) -> int:
        return int((1.0 - raw_x) * self.WIDTH)

    def _dot_row(self) -> int:
        return int(0.85 * self.HEIGHT)

    def _is_dot_pixel(self, frame: object, row: int, col: int) -> bool:
        """True if the pixel at (row, col) is a pure-green landmark dot."""
        import numpy as np  # noqa: PLC0415
        arr = np.asarray(frame)
        # BGR: dot = (0, 255, 0), text = (0, 255, 255)
        # Red channel (index 2) is 0 for a dot and 255 for text.
        return bool(arr[row, col, 1] > 0 and arr[row, col, 2] == 0)

    def test_non_mirror_places_landmark_at_raw_x(self) -> None:
        """Without mirroring, landmark at x=0.2 is drawn at the raw column."""
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = _make_state(landmarks=self._landmark_at(0.2))

        result = renderer.render(frame, state)
        row = self._dot_row()

        assert self._is_dot_pixel(result.frame, row, self._dot_col_raw(0.2)), (
            "Non-mirror: dot at raw x=0.2 must appear near col 128"
        )
        assert not self._is_dot_pixel(result.frame, row, self._dot_col_mirror(0.2)), (
            "Non-mirror: dot must NOT appear at the mirrored column"
        )

    def test_mirror_places_landmark_at_inverted_x(self) -> None:
        """With mirroring, landmark at raw x=0.2 is drawn at the inverted column."""
        renderer = OverlayRenderer(mirror_display=True)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = _make_state(landmarks=self._landmark_at(0.2))

        result = renderer.render(frame, state)
        row = self._dot_row()

        assert self._is_dot_pixel(result.frame, row, self._dot_col_mirror(0.2)), (
            "Mirror: dot at raw x=0.2 must appear at mirrored col (≈512)"
        )
        assert not self._is_dot_pixel(result.frame, row, self._dot_col_raw(0.2)), (
            "Mirror: dot must NOT appear at the raw column"
        )

    def test_mirror_does_not_affect_frame_dimensions(self) -> None:
        renderer = OverlayRenderer(mirror_display=True)
        frame = _make_frame(width=160, height=120)
        result = renderer.render(frame, _make_state())
        assert result.frame.shape == (120, 160, 3)

    def test_non_mirror_does_not_affect_frame_dimensions(self) -> None:
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=160, height=120)
        result = renderer.render(frame, _make_state())
        assert result.frame.shape == (120, 160, 3)


# ---------------------------------------------------------------------------
# Frame metadata is preserved
# ---------------------------------------------------------------------------

def test_render_preserves_timestamp_and_frame_id() -> None:
    renderer = OverlayRenderer(mirror_display=False)
    frame = FramePacket(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp_ms=99999,
        frame_id=42,
    )
    result = renderer.render(frame, _make_state())
    assert result.timestamp_ms == 99999
    assert result.frame_id == 42


# ---------------------------------------------------------------------------
# Original frame must not be mutated (ML pipeline reuse guard)
# ---------------------------------------------------------------------------

def test_render_does_not_mutate_original_frame() -> None:
    for mirror in (False, True):
        renderer = OverlayRenderer(mirror_display=mirror)
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = FramePacket(frame=original, timestamp_ms=0, frame_id=0)
        renderer.render(frame, _make_state(landmarks=[[0.5, 0.5, 0.0]]))
        assert original.sum() == 0, (
            f"Original frame must not be mutated (mirror={mirror})"
        )


# ---------------------------------------------------------------------------
# Invalid / edge-case landmarks are silently ignored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_landmarks", [
    None,
    "not a list",
    [None, None],
    [[0.5]],          # only one coordinate component
    [[]],             # empty point
])
def test_render_ignores_invalid_landmarks(bad_landmarks: object) -> None:
    renderer = OverlayRenderer(mirror_display=True)
    frame = _make_frame()
    state = _make_state(landmarks=bad_landmarks)
    result = renderer.render(frame, state)   # must not raise
    assert result.frame.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# Mirror symmetry: two landmarks mirrored about x=0.5 land on opposite pixels
# ---------------------------------------------------------------------------

def test_mirror_symmetry_of_two_landmarks() -> None:
    """Landmark at x=0.3 and x=0.7 should swap positions in mirror mode."""
    WIDTH, HEIGHT = 640, 480
    # Place landmark at y=0.85 so the dot is far below all text overlays.
    dot_row = int(0.85 * HEIGHT)
    renderer_mirror = OverlayRenderer(mirror_display=True)
    renderer_raw = OverlayRenderer(mirror_display=False)

    # BGR dot color (0,255,0): Red channel (idx 2) is 0 for a dot, 255 for text.
    def is_dot(frame: object, col: int) -> bool:
        import numpy as np  # noqa: PLC0415
        a = np.asarray(frame)
        return bool(a[dot_row, col, 1] > 0 and a[dot_row, col, 2] == 0)

    state_03 = _make_state(landmarks=[[0.3, 0.85, 0.0]])
    col_03_raw = int(0.3 * WIDTH)     # 192 in raw space
    col_03_mirror = int(0.7 * WIDTH)  # 448 in mirrored space

    # Non-mirror: dot at x=0.3 lands at raw col 192
    raw_result = renderer_raw.render(_make_frame(WIDTH, HEIGHT), state_03)
    assert is_dot(raw_result.frame, col_03_raw), (
        "Non-mirror: dot at raw x=0.3 must appear at col 192"
    )
    assert not is_dot(raw_result.frame, col_03_mirror), (
        "Non-mirror: dot must NOT appear at mirror col 448"
    )

    # Mirror: dot at raw x=0.3 lands at mirrored col 448
    mirror_result = renderer_mirror.render(_make_frame(WIDTH, HEIGHT), state_03)
    assert is_dot(mirror_result.frame, col_03_mirror), (
        "Mirror: dot at raw x=0.3 must appear at mirrored col 448"
    )
    assert not is_dot(mirror_result.frame, col_03_raw), (
        "Mirror: dot must NOT appear at raw col 192"
    )
