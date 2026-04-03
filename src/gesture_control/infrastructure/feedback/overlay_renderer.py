from __future__ import annotations

from typing import Any

import cv2

from gesture_control.domain.entities import FramePacket, RuntimeState

# Border colours (BGR).
# Absent  — grey:   gaze not directed toward camera; pipeline bypassed.
# Idle    — amber:  gaze detected but system not yet armed.
# Armed   — green:  system ready and accepting gesture commands.
_BORDER_ABSENT: tuple[int, int, int] = (128, 128, 128)
_BORDER_ACQUIRING: tuple[int, int, int] = (0, 165, 255)
_BORDER_ARMED: tuple[int, int, int] = (0, 200, 0)
_BORDER_THICKNESS = 4


class OverlayRenderer:
    """Renders non-blocking runtime diagnostics onto frames.

    Coordinate orientation policy
    --------------------------------
    MediaPipe landmark coordinates are always in the original (non-mirrored)
    camera coordinate space.  When ``mirror_display=True`` the canvas frame is
    flipped horizontally *before* landmarks are drawn so that the landmark dots
    remain correctly positioned on the mirrored image.  Text overlays are drawn
    after the flip so they remain readable.  The ML pipeline is never affected.

    Gaze border convention
    ----------------------
    A coloured border is drawn around the frame to give ambient feedback about
    the gaze-aware activation state:

    - **Grey**  (``gaze_permitted=False``): gaze absent; downstream pipeline
      bypassed entirely.
    - **Amber** (``gaze_permitted=True``, ``mode != "armed"``): gaze active but
      the activation gesture has not yet been performed; system is in IDLE or
      COOLDOWN.
    - **Green** (``gaze_permitted=True``, ``mode == "armed"``): system armed;
      gesture commands are accepted.
    """

    def __init__(self, *, mirror_display: bool = False) -> None:
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._mirror_display = mirror_display

    def render(self, frame: FramePacket, state: RuntimeState) -> FramePacket:
        # Flip the canvas for the display if mirror mode is active.  The flip
        # happens before landmarks are drawn so that x coordinates can be
        # inverted to match the mirrored pixel positions.
        if self._mirror_display:
            canvas = cv2.flip(frame.frame, 1)
        else:
            canvas = frame.frame.copy()

        self._draw_gaze_border(canvas, state)

        info = state.info

        self._draw_landmarks(canvas, info.get("landmarks"), mirror=self._mirror_display)

        line1 = f"mode={state.mode}"
        line2 = f"pred={state.last_label} conf={state.last_confidence:.2f}"
        line3 = (
            "window="
            f"{info.get('window_ready', False)} "
            f"ratio={float(info.get('valid_ratio', 0.0)):.2f}"
        )
        line4 = (
            f"latency_ms e2e={float(info.get('latency_e2e_ms', 0.0)):.1f} "
            f"infer={float(info.get('latency_infer_ms', 0.0)):.1f}"
        )
        line5 = f"fps={float(info.get('fps', 0.0)):.1f}"

        self._put(canvas, line1, 12, 24)
        self._put(canvas, line2, 12, 48)
        self._put(canvas, line3, 12, 72)
        self._put(canvas, line4, 12, 96)
        self._put(canvas, line5, 12, 120)

        return FramePacket(
            frame=canvas, timestamp_ms=frame.timestamp_ms, frame_id=frame.frame_id
        )

    def _draw_gaze_border(self, image: Any, state: RuntimeState) -> None:
        """Draw a coloured border reflecting the current gaze + activation state."""
        if not state.gaze_permitted:
            colour = _BORDER_ABSENT
        elif state.mode == "armed":
            colour = _BORDER_ARMED
        else:
            colour = _BORDER_ACQUIRING

        h, w = image.shape[:2]
        t = _BORDER_THICKNESS
        # Top and bottom bars
        image[:t, :] = colour
        image[h - t :, :] = colour
        # Left and right bars
        image[:, :t] = colour
        image[:, w - t :] = colour

    def _draw_landmarks(
        self, image: Any, landmarks: Any, *, mirror: bool = False
    ) -> None:
        if not isinstance(landmarks, list):
            return

        height, width = image.shape[:2]
        for point in landmarks:
            if not isinstance(point, list) or len(point) < 2:
                continue
            raw_x = float(point[0])
            # When the canvas is already flipped, invert the x coordinate so
            # the landmark dot lands on the correct side of the mirrored image.
            display_x = (1.0 - raw_x) if mirror else raw_x
            x = int(display_x * width)
            y = int(float(point[1]) * height)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    def _put(self, image: Any, text: str, x: int, y: int) -> None:
        cv2.putText(image, text, (x, y), self._font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
