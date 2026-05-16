from __future__ import annotations

import numpy as np
import pytest

from maestro.domain.entities import FramePacket, RuntimeState
from maestro.infrastructure.feedback.overlay_renderer import OverlayRenderer


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
        assert original.sum() == 0, f"Original frame must not be mutated (mirror={mirror})"


def test_minimal_preview_shows_border_hides_landmarks() -> None:
    """Minimal preview draws the gaze border but NOT landmarks."""
    renderer = OverlayRenderer(mirror_display=False, minimal_preview=True)
    state = RuntimeState(
        mode="armed",
        last_label="swipe_left",
        last_confidence=0.95,
        gaze_permitted=True,
        info={"landmarks": [[0.5, 0.5, 0.0]]},
    )

    result = renderer.render(_make_frame(width=160, height=240), state)

    # Gaze border IS drawn in minimal_preview mode (armed+gaze_permitted → green).
    assert tuple(int(v) for v in result.frame[1, 1]) != (0, 0, 0), (
        "expected gaze border pixel to be coloured, got black"
    )
    # Landmarks are NOT drawn — centre pixel stays dark.
    assert tuple(int(v) for v in result.frame[120, 80]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Invalid / edge-case landmarks are silently ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_landmarks",
    [
        None,
        "not a list",
        [None, None],
        [[0.5]],  # only one coordinate component
        [[]],  # empty point
    ],
)
def test_render_ignores_invalid_landmarks(bad_landmarks: object) -> None:
    renderer = OverlayRenderer(mirror_display=True)
    frame = _make_frame()
    state = _make_state(landmarks=bad_landmarks)
    result = renderer.render(frame, state)  # must not raise
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
    col_03_raw = int(0.3 * WIDTH)  # 192 in raw space
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
    assert not is_dot(mirror_result.frame, col_03_raw), "Mirror: dot must NOT appear at raw col 192"


# ---------------------------------------------------------------------------
# Gaze-aware border colour tests
# ---------------------------------------------------------------------------

# Border constants mirrored from the module under test so that the tests
# remain self-contained and do not depend on module-level name access.
_GREY = (128, 128, 128)
_AMBER = (0, 165, 255)
_GREEN = (0, 200, 0)
_LASER = (255, 140, 0)
_PEN = (180, 0, 200)
_THICKNESS = 4


def _border_pixel(frame: object, *, corner: str = "top-left") -> tuple[int, int, int]:
    """Return the BGR tuple of a border pixel in the requested corner region."""
    import numpy as np  # noqa: PLC0415

    a = np.asarray(frame)
    row, col = {
        "top-left": (1, 1),
        "bottom-right": (a.shape[0] - 2, a.shape[1] - 2),
    }[corner]
    return (int(a[row, col, 0]), int(a[row, col, 1]), int(a[row, col, 2]))


def _make_state_with_gaze(mode: str, gaze_permitted: bool) -> RuntimeState:
    return RuntimeState(
        mode=mode,
        last_label="unknown",
        last_confidence=0.0,
        gaze_permitted=gaze_permitted,
        info={},
    )


class TestGazeBorder:
    """Verify that the frame border reflects the gaze + activation state."""

    def _render(self, mode: str, gaze_permitted: bool) -> object:
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=100, height=100)
        state = _make_state_with_gaze(mode, gaze_permitted)
        return renderer.render(frame, state).frame

    def _render_with_info(self, mode: str, gaze_permitted: bool, info: dict) -> object:
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=100, height=100)
        state = RuntimeState(
            mode=mode,
            last_label="unknown",
            gaze_permitted=gaze_permitted,
            info=info,
        )
        return renderer.render(frame, state).frame

    # --- Absent state ---------------------------------------------------------

    def test_gaze_absent_renders_grey_border_top(self) -> None:
        """gaze_permitted=False → grey border regardless of mode."""
        frame = self._render("idle", gaze_permitted=False)
        assert _border_pixel(frame, corner="top-left") == _GREY

    def test_gaze_absent_renders_grey_border_bottom(self) -> None:
        frame = self._render("armed", gaze_permitted=False)
        assert _border_pixel(frame, corner="bottom-right") == _GREY

    def test_gaze_absent_overrides_armed_mode(self) -> None:
        """Even in ARMED mode, absent gaze must produce grey — not green."""
        frame = self._render("armed", gaze_permitted=False)
        assert _border_pixel(frame, corner="top-left") == _GREY

    # --- Acquiring / idle state -----------------------------------------------

    def test_gaze_active_idle_mode_renders_amber_border(self) -> None:
        """gaze_permitted=True + mode=idle → amber (gaze present, not yet armed)."""
        frame = self._render("idle", gaze_permitted=True)
        assert _border_pixel(frame, corner="top-left") == _AMBER

    def test_gaze_active_cooldown_mode_renders_amber_border(self) -> None:
        frame = self._render("cooldown", gaze_permitted=True)
        assert _border_pixel(frame, corner="top-left") == _AMBER

    # --- Armed state ----------------------------------------------------------

    def test_gaze_active_armed_mode_renders_green_border(self) -> None:
        """gaze_permitted=True + mode=armed → green (system ready)."""
        frame = self._render("armed", gaze_permitted=True)
        assert _border_pixel(frame, corner="top-left") == _GREEN

    def test_gaze_active_armed_border_on_all_edges(self) -> None:
        """All four border edges must carry the colour (spot-check top and bottom)."""
        frame = self._render("armed", gaze_permitted=True)
        assert _border_pixel(frame, corner="top-left") == _GREEN
        assert _border_pixel(frame, corner="bottom-right") == _GREEN

    # --- Border does not corrupt interior pixels ------------------------------

    def test_border_does_not_corrupt_interior(self) -> None:
        """Border drawing must only colour edge rows/cols, not the interior.

        Uses a 400×200 canvas so text (starting at x=12, max ~310 px wide) does
        not reach the sampled region (columns 350-390, rows 6-16 — well above
        the first text baseline at y=24 and far right of all text columns).
        """
        import numpy as np  # noqa: PLC0415

        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=400, height=200)
        state = _make_state_with_gaze("idle", gaze_permitted=True)
        rendered = renderer.render(frame, state).frame
        # Rows just inside the top border, columns far to the right of all text.
        patch = np.asarray(rendered)[_THICKNESS + 2 : _THICKNESS + 12, 350:390]
        assert patch.sum() == 0, "Interior region far from text must remain unmodified"

    # --- Default gaze_permitted=True is backward-compatible ------------------

    def test_default_state_renders_amber_border(self) -> None:
        """RuntimeState with default gaze_permitted=True and mode='idle' → amber."""
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=100, height=100)
        state = RuntimeState(mode="idle", last_label="unknown", last_confidence=0.0)
        result = renderer.render(frame, state).frame
        assert _border_pixel(result, corner="top-left") == _AMBER

    # --- Continuous gesture states (laser/pen) --------------------------------

    def test_laser_active_renders_sky_blue_border(self) -> None:
        """laser_active=True in info → sky blue border overrides mode."""
        frame = self._render_with_info("idle", gaze_permitted=True, info={"laser_active": True})
        assert _border_pixel(frame, corner="top-left") == _LASER

    def test_laser_active_overrides_armed_mode(self) -> None:
        """laser_active=True must override even armed mode → sky blue."""
        frame = self._render_with_info("armed", gaze_permitted=True, info={"laser_active": True})
        assert _border_pixel(frame, corner="top-left") == _LASER

    def test_pen_active_renders_violet_border(self) -> None:
        """pen_active=True in info → violet border overrides mode."""
        frame = self._render_with_info("idle", gaze_permitted=True, info={"pen_active": True})
        assert _border_pixel(frame, corner="top-left") == _PEN

    def test_pen_active_overrides_armed_mode(self) -> None:
        """pen_active=True must override even armed mode → violet."""
        frame = self._render_with_info("armed", gaze_permitted=True, info={"pen_active": True})
        assert _border_pixel(frame, corner="top-left") == _PEN

    def test_gaze_absent_overrides_laser_active(self) -> None:
        """gaze_permitted=False takes priority over laser_active."""
        frame = self._render_with_info(
            "armed",
            gaze_permitted=False,
            info={"laser_active": True},
        )
        assert _border_pixel(frame, corner="top-left") == _GREY

    def test_gaze_absent_overrides_pen_active(self) -> None:
        """gaze_permitted=False takes priority over pen_active."""
        frame = self._render_with_info(
            "armed",
            gaze_permitted=False,
            info={"pen_active": True},
        )
        assert _border_pixel(frame, corner="top-left") == _GREY


# ---------------------------------------------------------------------------
# Phase 4: Mode badge and command preview
# ---------------------------------------------------------------------------


def _has_any_non_black_pixel_in_row(frame: np.ndarray, y: int, *, tolerance: int = 5) -> bool:
    """Return True if any pixel in the frame row at y-offset is brighter than black."""
    row_start = max(0, y - tolerance)
    row_end = min(frame.shape[0], y + tolerance)
    patch = frame[row_start:row_end, :]
    return bool(patch.max() > 0)


class TestModeBadge:
    """Verify that the mode badge is rendered when active_mode is set."""

    WIDTH, HEIGHT = 640, 480

    def _make_desktop_state(self, active_mode: str, label: str = "unknown") -> RuntimeState:
        return RuntimeState(
            mode="armed",
            last_label=label,
            last_confidence=0.9,
            active_mode=active_mode,
        )

    def test_mode_badge_rendered_when_active_mode_not_unknown(self) -> None:
        """A non-'unknown' active_mode must produce visible text near y=18."""
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = self._make_desktop_state("desktop\u00b7workspace")
        result = renderer.render(frame, state)
        # Row 18 (badge y) should contain non-black pixels (the badge text).
        assert _has_any_non_black_pixel_in_row(result.frame, 18), (
            "Mode badge text must produce visible pixels near row 18"
        )

    def test_no_mode_badge_when_active_mode_is_unknown(self) -> None:
        """Default 'unknown' active_mode must NOT render a white badge row.

        The badge is drawn in white (B=255, G=255, R=255 in BGR).  All regular
        HUD text uses yellow (B=0, G=255, R=255).  Checking that the blue
        channel is 0 in the badge row area confirms that no white badge text
        was rendered, even if yellow HUD text overlaps the same y range.
        """
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = RuntimeState(
            mode="idle",
            last_label="unknown",
            last_confidence=0.0,
            active_mode="unknown",
        )
        result = renderer.render(frame, state)
        # Badge text colour is white (B=255).  Yellow HUD text has B=0.
        # Confirm no white pixels in rows 14-22 (blue channel must stay 0).
        patch_blue = np.asarray(result.frame)[14:22, 8:300, 0]  # B channel
        assert patch_blue.max() == 0, (
            "Row 18 must have no white pixels (B=255) when active_mode='unknown'; "
            "only yellow HUD text (B=0) is permitted there"
        )

    def test_command_map_accepted_in_constructor(self) -> None:
        """OverlayRenderer must accept command_map param without error."""
        cmap = {"stop_sign": "mission_control", "unknown": "no_action"}
        renderer = OverlayRenderer(mirror_display=False, command_map=cmap)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = self._make_desktop_state("desktop\u00b7workspace", label="stop_sign")
        result = renderer.render(frame, state)
        assert result.frame.shape == (self.HEIGHT, self.WIDTH, 3)

    def test_render_succeeds_without_command_map(self) -> None:
        """OverlayRenderer with no command_map must render without error."""
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = self._make_desktop_state("desktop\u00b7workspace")
        result = renderer.render(frame, state)
        assert result.frame.shape == (self.HEIGHT, self.WIDTH, 3)

    def test_phase4_gate_desktop_workspace_mission_control(self) -> None:
        """Gate: overlay displays [desktop·workspace] → mission_control.

        A synthetic RuntimeState with active_mode='desktop·workspace' and
        last_label='stop_sign', rendered through a renderer with the
        desktop command map, must produce a non-empty canvas containing
        visible pixels in both the badge row (y≈18) and pred ring row (y≈72).
        """
        cmap = {
            "stop_sign": "mission_control",
            "unknown": "no_action",
            "swiping_up": "scroll_forward",
        }
        renderer = OverlayRenderer(mirror_display=False, command_map=cmap)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = RuntimeState(
            mode="armed",
            last_label="stop_sign",
            last_confidence=0.97,
            active_mode="desktop\u00b7workspace",
        )
        result = renderer.render(frame, state)
        assert result.frame.shape == (self.HEIGHT, self.WIDTH, 3)
        # Badge row must be lit (mode badge present).
        assert _has_any_non_black_pixel_in_row(result.frame, 18), (
            "Phase 4 gate: badge row must contain visible pixels"
        )
        # Pred ring must exist (non-black background in the pred line area).
        pred_y = 48 + 24  # y_offset=24 because active_mode != 'unknown'
        assert _has_any_non_black_pixel_in_row(result.frame, pred_y), (
            "Phase 4 gate: pred ring must contain visible pixels"
        )


class TestPredRingColours:
    """Verify pred ring background colour reflects the mode state."""

    WIDTH, HEIGHT = 640, 480

    def _render_ring(self, mode: str, active_mode: str = "desktop\u00b7workspace") -> np.ndarray:
        renderer = OverlayRenderer(mirror_display=False)
        frame = _make_frame(width=self.WIDTH, height=self.HEIGHT)
        state = RuntimeState(
            mode=mode,
            last_label="unknown",
            last_confidence=0.0,
            active_mode=active_mode,
        )
        return renderer.render(frame, state).frame

    def test_armed_ring_is_dark_green(self) -> None:
        frame = self._render_ring("armed")
        pred_y = 48 + 24
        # The ring for 'armed' is a dark-green band; check green > 0 in that row.
        row_start, row_end = pred_y - 18, pred_y + 6
        patch = frame[row_start:row_end, 50:200]
        # Dark green has G channel set; the band should not be fully black.
        assert patch[:, :, 1].max() > 0, "Armed pred ring must have G channel > 0"

    def test_ring_renders_without_error_for_all_known_modes(self) -> None:
        for mode in ("idle", "armed", "cooldown"):
            frame = self._render_ring(mode)
            assert frame.shape == (self.HEIGHT, self.WIDTH, 3)
