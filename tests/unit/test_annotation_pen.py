"""Unit tests for AnnotationPenTracker.

All OS side-effects (pyautogui, keyboard calls) are patched through a mock
KeyboardAdapter, so the tests run cleanly in headless / CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

from maestro.pipeline.annotation_pen import AnnotationPenTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PEN_LABELS = frozenset({"point_two"})

# 21 landmarks; index 8 is the index fingertip.
_LANDMARKS_CENTER: list[list[float]] = [[0.5, 0.5, 0.0] for _ in range(21)]
_LANDMARKS_TOP_RIGHT: list[list[float]] = [[0.8, 0.2, 0.0] for _ in range(21)]


def _mock_keyboard() -> MagicMock:
    kb = MagicMock()
    kb.screen_size.return_value = (1920, 1080)
    return kb


def _tracker(
    mirror_x: bool = True,
    use_command_key: bool = False,
) -> tuple[AnnotationPenTracker, MagicMock]:
    kb = _mock_keyboard()
    # Pin to Windows/Linux path by default (ctrl+p / ctrl+a) so these tests
    # are platform-independent.  Pass use_command_key=True to exercise the
    # macOS Command+P / Command+A path.
    on_keys = ("command", "p") if use_command_key else ("ctrl", "p")
    off_keys = ("command", "a") if use_command_key else ("ctrl", "a")
    tracker = AnnotationPenTracker(
        keyboard=kb,
        pointing_labels=_PEN_LABELS,
        mirror_x=mirror_x,
        pen_on_keys=on_keys,
        pen_off_keys=off_keys,
    )
    return tracker, kb


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerInitialState:
    def test_starts_inactive(self):
        tracker, _ = _tracker()
        assert tracker.active is False

    def test_non_pen_label_stays_inactive(self):
        tracker, kb = _tracker()
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()

    def test_point_one_does_not_activate_pen(self):
        """point_one is for the laser pointer, not the annotation pen."""
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: activation
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerActivation:
    def test_point_two_activates_pen(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("ctrl", "p")

    def test_pen_on_key_sent_only_on_entry(self):
        """Ctrl+P must fire once on first pen frame, not on subsequent frames."""
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        ctrl_p_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "p")]
        assert len(ctrl_p_calls) == 1

    def test_invalid_landmarks_prevents_activation(self):
        tracker, kb = _tracker()
        tracker.update("point_two", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()

    def test_activation_does_not_reset_cursor_to_centre(self):
        """Annotation pen must NOT move the cursor to screen centre on activation.

        Unlike the laser pointer, the pen stroke must start exactly at the
        user's fingertip position so that the drawn mark appears where the
        user intends.  Positioning is done via mouse_down, not move_mouse.
        """
        tracker, kb = _tracker()
        kb.screen_size.return_value = (1920, 1080)
        lm_off_centre = [[0.8, 0.2, 0.0]] * 21
        tracker.update("point_two", lm_off_centre, valid=True)
        # move_mouse must never be called — only mouse_down for the pen.
        kb.move_mouse.assert_not_called()
        kb.mouse_down.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: mouse tracking while active
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerMouseTracking:
    def test_mouse_down_called_on_first_active_frame(self):
        """First frame: position and press — PowerPoint starts recording the stroke."""
        tracker, kb = _tracker(mirror_x=False)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        kb.mouse_down.assert_called_once()
        kb.mouse_drag_to.assert_not_called()
        kb.move_mouse.assert_not_called()

    def test_drag_called_on_subsequent_frames(self):
        """Sustained frames send kCGEventLeftMouseDragged (not kCGEventMouseMoved)."""
        tracker, kb = _tracker(mirror_x=False)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # mouse_down
        kb.mouse_drag_to.reset_mock()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        kb.mouse_drag_to.assert_called_once()
        kb.move_mouse.assert_not_called()  # plain moveTo must never be used
        kb.mouse_down.assert_called_once()  # not called again

    def test_mirror_x_flips_horizontal_coordinate(self):
        tracker_mirrored, kb_m = _tracker(mirror_x=True)
        tracker_straight, kb_s = _tracker(mirror_x=False)
        kb_m.screen_size.return_value = (1000, 1000)
        kb_s.screen_size.return_value = (1000, 1000)

        lm = [[0.8, 0.5, 0.0]] * 21

        tracker_mirrored.update("point_two", lm, valid=True)
        tracker_straight.update("point_two", lm, valid=True)

        # First frame uses mouse_down(x, y) — check those coordinates.
        x_mirrored = kb_m.mouse_down.call_args[0][0]
        x_straight = kb_s.mouse_down.call_args[0][0]

        assert x_mirrored < x_straight

    def test_y_coordinate_maps_correctly(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1920, 1080)
        lm = [[0.5, 0.25, 0.0]] * 21
        tracker.update("point_two", lm, valid=True)
        # First frame: mouse_down(x, y)
        _, y = kb.mouse_down.call_args[0]
        assert y == 0

    def test_center_point_remains_stable_after_gain(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        x, y = kb.mouse_down.call_args[0]
        assert x == 500
        assert y == 500

    def test_horizontal_pointer_gain_expands_displacement(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        lm = [[0.6, 0.5, 0.0]] * 21
        tracker.update("point_two", lm, valid=True)
        x, _ = kb.mouse_down.call_args[0]
        assert x == int((0.5 + ((0.6 - 0.5) * 2.4)) * 1000)

    def test_vertical_pointer_gain_expands_displacement(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        lm = [[0.5, 0.6, 0.0]] * 21
        tracker.update("point_two", lm, valid=True)
        _, y = kb.mouse_down.call_args[0]
        assert y == int((0.5 + ((0.6 - 0.5) * 2.2)) * 1000)

    def test_coordinates_clamped_to_screen_bounds(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1920, 1080)
        lm = [[1.0, 1.0, 0.0]] * 21
        tracker.update("point_two", lm, valid=True)
        # First frame: mouse_down(x, y)
        x, y = kb.mouse_down.call_args[0]
        assert x == 1919
        assert y == 1079


# ---------------------------------------------------------------------------
# Tests: deactivation
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerDeactivation:
    def test_unknown_label_deactivates(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")

    def test_mouse_up_before_pen_off_hotkey(self):
        """mouse_up must be called before the revert hotkey so the stroke ends cleanly."""
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        kb.reset_mock()
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        call_names = [c[0] for c in kb.method_calls]
        assert "mouse_up" in call_names
        assert "press_hotkey" in call_names
        assert call_names.index("mouse_up") < call_names.index("press_hotkey")

    def test_mouse_up_called_once_on_exit(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.mouse_up.assert_called_once()

    def test_pen_off_key_sent_only_once_on_exit(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        ctrl_a_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "a")]
        assert len(ctrl_a_calls) == 1

    def test_invalid_hand_deactivates(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_two", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")
        kb.mouse_up.assert_called_once()

    def test_explicit_deactivate_sends_pen_off_key(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.deactivate()
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")
        kb.mouse_up.assert_called_once()

    def test_explicit_deactivate_when_already_inactive_is_noop(self):
        tracker, kb = _tracker()
        tracker.deactivate()
        kb.press_hotkey.assert_not_called()
        kb.mouse_up.assert_not_called()
        assert tracker.active is False

    def test_no_mouse_move_after_deactivation(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.reset_mock()
        kb.mouse_drag_to.reset_mock()
        kb.mouse_down.reset_mock()
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.assert_not_called()
        kb.mouse_drag_to.assert_not_called()
        kb.mouse_down.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: re-activation after deactivation
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerReactivation:
    def test_can_reactivate_after_deactivation(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.reset_mock()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("ctrl", "p")

    def test_pen_on_key_fires_again_after_neutral(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # activate
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)  # deactivate
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # re-activate
        ctrl_p_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "p")]
        assert len(ctrl_p_calls) == 2

    def test_mouse_down_fires_again_after_neutral(self):
        """After re-activation, mouse_down starts a fresh stroke."""
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # activate → mouse_down
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)  # deactivate → mouse_up
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # re-activate → mouse_down
        assert kb.mouse_down.call_count == 2
        assert kb.mouse_up.call_count == 1


# ---------------------------------------------------------------------------
# Tests: macOS Command+P / Command+A mode
# ---------------------------------------------------------------------------


class TestAnnotationPenTrackerMacOS:
    """Tests for the macOS ⌘+P / ⌘+A annotation pen activation mode."""

    def test_activation_sends_command_p(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("command", "p")

    def test_command_p_sent_only_on_entry(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        cmd_p_calls = [c for c in kb.press_hotkey.call_args_list if c == call("command", "p")]
        assert len(cmd_p_calls) == 1

    def test_deactivation_sends_command_a(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("command", "a")

    def test_explicit_deactivate_sends_command_a(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        tracker.deactivate()
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("command", "a")

    def test_reactivation_sends_command_p_again(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # activate
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)  # deactivate
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # re-activate
        cmd_p_calls = [c for c in kb.press_hotkey.call_args_list if c == call("command", "p")]
        assert len(cmd_p_calls) == 2

    def test_mouse_still_tracked_while_active(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # mouse_down
        kb.mouse_down.reset_mock()
        kb.mouse_drag_to.reset_mock()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)  # mouse_drag_to
        kb.mouse_drag_to.assert_called_once()
        kb.move_mouse.assert_not_called()
        kb.mouse_down.assert_not_called()

    def test_invalid_landmarks_prevents_activation(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_two", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()
