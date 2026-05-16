from __future__ import annotations

from typing import Any

import cv2

from maestro.domain.entities import FramePacket, RuntimeState

# Border colours (BGR).
# Absent   — grey:   gaze not directed toward camera; pipeline bypassed.
# Idle     — amber:  system not yet armed (IDLE or post-cooldown wait).
# Armed    — green:  system ready and accepting gesture commands.
# Cooldown — amber:  same as idle colour; command just dispatched.
# Laser    — sky blue: laser pointer tracker actively driving cursor.
# Pen      — violet: annotation pen tracker actively drawing.
_BORDER_ABSENT: tuple[int, int, int] = (128, 128, 128)
_BORDER_ACQUIRING: tuple[int, int, int] = (0, 165, 255)
_BORDER_ARMED: tuple[int, int, int] = (0, 200, 0)
_BORDER_LASER: tuple[int, int, int] = (255, 140, 0)  # sky blue (BGR)
_BORDER_PEN: tuple[int, int, int] = (180, 0, 200)   # violet (BGR)
_BORDER_THICKNESS = 4

# Prediction ring — a coloured background rectangle drawn behind the
# gesture prediction line to give immediate mode-state feedback.
_RING_ARMED: tuple[int, int, int] = (0, 60, 0)    # dark green background
_RING_COOLDOWN: tuple[int, int, int] = (0, 55, 90)  # dark amber background
_RING_LASER: tuple[int, int, int] = (60, 30, 0)   # dark blue background
_RING_PEN: tuple[int, int, int] = (55, 0, 60)     # dark violet background
_RING_DEFAULT: tuple[int, int, int] = (30, 30, 30)  # dark grey background

# Text colours (BGR).
_TEXT_BADGE: tuple[int, int, int] = (255, 255, 255)  # white — mode badge
_TEXT_OVERLAY: tuple[int, int, int] = (0, 255, 255)  # yellow — HUD lines


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

    - **Grey**      (``gaze_permitted=False``): gaze absent; pipeline bypassed.
    - **Amber**     (``gaze_permitted=True``, ``mode != "armed"``): gaze active
      but not yet armed; system is in IDLE or COOLDOWN.
    - **Green**     (``gaze_permitted=True``, ``mode == "armed"``): system armed;
      discrete gesture commands are accepted.
    - **Sky blue**  (``laser_active=True`` in ``state.info``): laser pointer
      tracker is actively driving the cursor.
    - **Violet**    (``pen_active=True`` in ``state.info``): annotation pen
      tracker is actively drawing.
    """

    def __init__(
        self,
        *,
        mirror_display: bool = False,
        command_map: dict[str, str] | None = None,
        output_rgb: bool = False,
        minimal_preview: bool = False,
    ) -> None:
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._mirror_display = mirror_display
        # command_map: gesture label → logical command ID for display lookup.
        # When None the logical command column is omitted from the pred line.
        self._command_map: dict[str, str] = command_map or {}
        # When True the rendered canvas is converted from OpenCV's native BGR
        # to RGB before being returned.  Set this to True when the consumer
        # is a Qt widget (QImage.Format_RGB888) so the channel reversal is
        # done once here rather than on every display update.
        self._output_rgb = output_rgb
        self._minimal_preview = minimal_preview

    def render(self, frame: FramePacket, state: RuntimeState) -> FramePacket:
        # Flip the canvas for the display if mirror mode is active.  The flip
        # happens before landmarks are drawn so that x coordinates can be
        # inverted to match the mirrored pixel positions.
        if self._mirror_display:
            canvas = cv2.flip(frame.frame, 1)
        else:
            canvas = frame.frame.copy()

        info = state.info

        if self._minimal_preview:
            self._draw_gaze_border(canvas, state)
            self._draw_minimal_gesture_label(canvas, state.last_label)
        else:
            self._draw_gaze_border(canvas, state)
            self._draw_landmarks(canvas, info.get("landmarks"), mirror=self._mirror_display)

            # --- Mode badge -------------------------------------------------
            # Render "[active_mode]" at the very top of the frame when a mode has
            # been set (i.e. it is not the default sentinel "unknown").
            y_offset = 0
            if state.active_mode != "unknown":
                badge_text = f"[{state.active_mode}]"
                self._put(canvas, badge_text, 12, 18, color=_TEXT_BADGE)
                y_offset = 24  # push HUD lines down to avoid overlapping the badge

            # --- Prediction line with logical command preview ---------------
            logical_cmd = self._command_map.get(state.last_label, "")
            if logical_cmd and logical_cmd != "no_action":
                line2 = (
                    f"pred={state.last_label}  → {logical_cmd}  conf={state.last_confidence:.2f}"
                )
            else:
                line2 = f"pred={state.last_label} conf={state.last_confidence:.2f}"

            # Draw a coloured background rectangle (the "hue ring") behind the
            # prediction line so the mode-state colour is immediately visible
            # without obscuring the rest of the frame content.
            pred_y = 48 + y_offset
            self._draw_pred_ring(canvas, state.mode, y=pred_y, info=state.info)

            line1 = f"mode={state.mode}"
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

            self._put(canvas, line1, 12, 24 + y_offset)
            self._put(canvas, line2, 12, pred_y)
            self._put(canvas, line3, 12, 72 + y_offset)
            self._put(canvas, line4, 12, 96 + y_offset)
            self._put(canvas, line5, 12, 120 + y_offset)

        if self._output_rgb:
            canvas = canvas[..., ::-1]

        return FramePacket(frame=canvas, timestamp_ms=frame.timestamp_ms, frame_id=frame.frame_id)

    def _draw_gaze_border(self, image: Any, state: RuntimeState) -> None:
        """Draw a coloured border reflecting the current gaze + activation state."""
        if not state.gaze_permitted:
            colour = _BORDER_ABSENT
        elif state.info.get("laser_active"):
            colour = _BORDER_LASER
        elif state.info.get("pen_active"):
            colour = _BORDER_PEN
        elif state.mode == "armed":
            colour = _BORDER_ARMED
        else:
            colour = _BORDER_ACQUIRING

        h, w = image.shape[:2]
        # Scale thickness with source-frame size so the border remains clearly
        # visible after the frame is downscaled for the floating preview.
        t = max(_BORDER_THICKNESS, int(round(min(h, w) * 0.012)))
        # Top and bottom bars
        image[:t, :] = colour
        image[h - t :, :] = colour
        # Left and right bars
        image[:, :t] = colour
        image[:, w - t :] = colour

    def _draw_minimal_gesture_label(self, image: Any, label: str) -> None:
        """Draw only the current gesture label for the floating self preview."""
        if not label or label == "unknown":
            return

        text = label.replace("_", " ").upper()
        height, width = image.shape[:2]

        # Scale font so the text looks large at the *displayed* size after the
        # frame is downscaled for the preview widget.  Target ~18 % of height.
        font_scale = max(1.8, (height * 0.18) / 28.0)
        thickness = max(3, int(font_scale * 2))

        # Measure text to set footer height and baseline position accurately.
        (text_w, text_h), baseline = cv2.getTextSize(text, self._font, font_scale, thickness)
        footer_height = text_h + baseline + 20
        y0 = max(0, height - footer_height)
        image[y0:height, 0:width] = (20, 20, 20)

        # Centre the label horizontally; keep a safe bottom margin.
        x = max(8, (width - text_w) // 2)
        y = height - baseline - 8
        cv2.putText(
            image,
            text,
            (x, y),
            self._font,
            font_scale,
            _TEXT_BADGE,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_pred_ring(
        self,
        image: Any,
        mode: str,
        *,
        y: int,
        info: dict | None = None,
    ) -> None:
        """Draw a coloured background rectangle behind the prediction text line."""
        if info is not None and info.get("laser_active"):
            bg = _RING_LASER
        elif info is not None and info.get("pen_active"):
            bg = _RING_PEN
        elif mode == "armed":
            bg = _RING_ARMED
        elif mode == "cooldown":
            bg = _RING_COOLDOWN
        else:
            bg = _RING_DEFAULT
        h, w = image.shape[:2]
        # The ring spans the full width and covers one text line (~22 px).
        y0 = max(0, y - 18)
        y1 = min(h, y + 6)
        image[y0:y1, 0:w] = bg

    def _draw_landmarks(self, image: Any, landmarks: Any, *, mirror: bool = False) -> None:
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

    def _put(
        self,
        image: Any,
        text: str,
        x: int,
        y: int,
        *,
        color: tuple[int, int, int] = _TEXT_OVERLAY,
    ) -> None:
        cv2.putText(image, text, (x, y), self._font, 0.6, color, 2, cv2.LINE_AA)
