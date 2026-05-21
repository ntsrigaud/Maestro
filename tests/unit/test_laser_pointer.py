"""Unit tests for LaserPointerTracker.

All OS side-effects (pyautogui, keyboard calls) are patched through a mock
KeyboardAdapter, so the tests run cleanly in headless / CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

from maestro.pipeline.laser_pointer import LaserPointerTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POINTING_LABELS = frozenset({"point_one", "point_two"})

# 21 landmarks; index 8 is the index fingertip.
_LANDMARKS_CENTER: list[list[float]] = [[0.5, 0.5, 0.0] for _ in range(21)]
_LANDMARKS_TOP_RIGHT: list[list[float]] = [[0.8, 0.2, 0.0] for _ in range(21)]


def _mock_keyboard() -> MagicMock:
    kb = MagicMock()
    kb.screen_size.return_value = (1920, 1080)
    return kb


def _tracker(
    mirror_x: bool = True, use_command_key: bool = False
) -> tuple[LaserPointerTracker, MagicMock]:
    kb = _mock_keyboard()
    # Pin to Windows/Linux path by default (ctrl+l / ctrl+a) so these tests
    # are platform-independent.  Pass use_command_key=True to exercise the
    # macOS Command+L / Command+A path.
    on_keys = ("command", "l") if use_command_key else ("ctrl", "l")
    off_keys = ("command", "a") if use_command_key else ("ctrl", "a")
    tracker = LaserPointerTracker(
        keyboard=kb,
        pointing_labels=_POINTING_LABELS,
        mirror_x=mirror_x,
        laser_on_keys=on_keys,
        laser_off_keys=off_keys,
    )
    return tracker, kb


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerInitialState:
    def test_starts_inactive(self):
        tracker, _ = _tracker()
        assert tracker.active is False

    def test_non_pointing_label_stays_inactive(self):
        tracker, kb = _tracker()
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: activation
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerActivation:
    def test_pointing_gesture_activates(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("ctrl", "l")

    def test_ctrl_l_sent_only_on_entry(self):
        """Ctrl+L must fire once on first pointing frame, not on subsequent frames."""
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        ctrl_l_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "l")]
        assert len(ctrl_l_calls) == 1

    def test_point_two_also_activates(self):
        tracker, kb = _tracker()
        tracker.update("point_two", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("ctrl", "l")

    def test_invalid_landmarks_prevents_activation(self):
        tracker, kb = _tracker()
        tracker.update("point_one", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()

    def test_cursor_resets_to_screen_centre_on_activation(self):
        """First move_mouse call on activation must land at screen centre.

        Uses an off-centre landmark so the centre-reset and subsequent
        fingertip moves produce different coordinates — making the assertion
        unambiguous.
        """
        tracker, kb = _tracker()
        kb.screen_size.return_value = (1920, 1080)
        lm_off_centre = [[0.8, 0.2, 0.0]] * 21
        tracker.update("point_one", lm_off_centre, valid=True)
        first_move = kb.move_mouse.call_args_list[0]
        assert first_move == call(960, 540), (
            "Laser pointer must reset cursor to screen centre (960, 540) on activation"
        )

    def test_cursor_resets_to_centre_on_every_reactivation(self):
        """Screen-centre reset fires each time the laser is re-activated."""
        tracker, kb = _tracker()
        kb.screen_size.return_value = (1920, 1080)
        lm_off_centre = [[0.8, 0.2, 0.0]] * 21
        # First activation then deactivation.
        tracker.update("point_one", lm_off_centre, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.reset_mock()
        # Second activation — centre-reset must fire again.
        tracker.update("point_one", lm_off_centre, valid=True)
        first_move_reactivation = kb.move_mouse.call_args_list[0]
        assert first_move_reactivation == call(960, 540)


# ---------------------------------------------------------------------------
# Tests: mouse tracking while active
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerMouseTracking:
    def test_move_mouse_called_each_pointing_frame(self):
        tracker, kb = _tracker(mirror_x=False)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # activates
        kb.move_mouse.reset_mock()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.assert_called_once()

    def test_mirror_x_flips_horizontal_coordinate(self):
        """With mirror_x=True, x=0.8 norm → screen_x near left (small pixel value)."""
        tracker_mirrored, kb_m = _tracker(mirror_x=True)
        tracker_straight, kb_s = _tracker(mirror_x=False)
        kb_m.screen_size.return_value = (1000, 1000)
        kb_s.screen_size.return_value = (1000, 1000)

        lm = [[0.8, 0.5, 0.0]] * 21

        tracker_mirrored.update("point_one", lm, valid=True)
        tracker_straight.update("point_one", lm, valid=True)

        x_mirrored = kb_m.move_mouse.call_args[0][0]
        x_straight = kb_s.move_mouse.call_args[0][0]

        # mirror_x flips the x-axis: a landmark at x=0.8 (right in camera frame)
        # should map to a smaller screen_x when mirrored (i.e. left of screen).
        assert x_mirrored < x_straight

    def test_y_coordinate_maps_correctly(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1920, 1080)
        lm = [[0.5, 0.25, 0.0]] * 21
        tracker.update("point_one", lm, valid=True)
        _, y = kb.move_mouse.call_args[0]
        assert y == 0

    def test_center_point_remains_stable_after_gain(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        x, y = kb.move_mouse.call_args[0]
        assert x == 500
        assert y == 500

    def test_horizontal_pointer_gain_expands_displacement(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        lm = [[0.6, 0.5, 0.0]] * 21
        tracker.update("point_one", lm, valid=True)
        x, _ = kb.move_mouse.call_args[0]
        assert x == int((0.5 + ((0.6 - 0.5) * 2.4)) * 1000)

    def test_vertical_pointer_gain_expands_displacement(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1000, 1000)
        lm = [[0.5, 0.6, 0.0]] * 21
        tracker.update("point_one", lm, valid=True)
        _, y = kb.move_mouse.call_args[0]
        assert y == int((0.5 + ((0.6 - 0.5) * 2.2)) * 1000)

    def test_coordinates_clamped_to_screen_bounds(self):
        tracker, kb = _tracker(mirror_x=False)
        kb.screen_size.return_value = (1920, 1080)
        # Landmark at exactly x=1.0 → screen_x = 1920, clamped to 1919.
        lm = [[1.0, 1.0, 0.0]] * 21
        tracker.update("point_one", lm, valid=True)
        x, y = kb.move_mouse.call_args[0]
        assert x == 1919
        assert y == 1079


# ---------------------------------------------------------------------------
# Tests: deactivation
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerDeactivation:
    def test_unknown_label_deactivates(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")

    def test_ctrl_a_sent_only_once_on_exit(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        ctrl_a_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "a")]
        assert len(ctrl_a_calls) == 1

    def test_invalid_hand_deactivates(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_one", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")

    def test_explicit_deactivate_sends_ctrl_a(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.deactivate()
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("ctrl", "a")

    def test_explicit_deactivate_when_already_inactive_is_noop(self):
        tracker, kb = _tracker()
        tracker.deactivate()  # was never active
        kb.press_hotkey.assert_not_called()
        assert tracker.active is False

    def test_no_mouse_move_after_deactivation(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.reset_mock()
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: re-activation after deactivation
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerReactivation:
    def test_can_reactivate_after_deactivation(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        kb.reset_mock()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("ctrl", "l")

    def test_ctrl_l_fires_again_after_neutral(self):
        tracker, kb = _tracker()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # activate
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)  # deactivate
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # re-activate
        ctrl_l_calls = [c for c in kb.press_hotkey.call_args_list if c == call("ctrl", "l")]
        assert len(ctrl_l_calls) == 2


# ---------------------------------------------------------------------------
# Tests: macOS Command+L / Command+A mode
# ---------------------------------------------------------------------------


class TestLaserPointerTrackerMacOS:
    """Tests for the macOS ⌘+L / ⌘+A laser pointer activation mode."""

    def test_activation_sends_command_l(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is True
        kb.press_hotkey.assert_any_call("command", "l")

    def test_command_l_sent_only_on_entry(self):
        """Command+L must fire once on first pointing frame, not on subsequent frames."""
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        cmd_l_calls = [c for c in kb.press_hotkey.call_args_list if c == call("command", "l")]
        assert len(cmd_l_calls) == 1

    def test_deactivation_sends_command_a(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("command", "a")

    def test_explicit_deactivate_sends_command_a(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        tracker.deactivate()
        assert tracker.active is False
        kb.press_hotkey.assert_any_call("command", "a")

    def test_reactivation_sends_command_l_again(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # activate
        tracker.update("unknown", _LANDMARKS_CENTER, valid=True)  # deactivate
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # re-activate
        cmd_l_calls = [c for c in kb.press_hotkey.call_args_list if c == call("command", "l")]
        assert len(cmd_l_calls) == 2

    def test_mouse_still_tracked_while_active(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)  # activates + moves
        kb.move_mouse.reset_mock()
        tracker.update("point_one", _LANDMARKS_CENTER, valid=True)
        kb.move_mouse.assert_called_once()

    def test_invalid_landmarks_prevents_activation(self):
        tracker, kb = _tracker(use_command_key=True)
        tracker.update("point_one", None, valid=False)
        assert tracker.active is False
        kb.press_hotkey.assert_not_called()
