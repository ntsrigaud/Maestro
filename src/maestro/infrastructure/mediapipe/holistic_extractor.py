"""MediaPipe Holistic landmark extractor for the graph-schema-v1 pipeline.

Uses the mediapipe >= 0.10 task-based ``HolisticLandmarker`` API to extract
67 normalised keypoints (right hand 0–20, left hand 21–41, pose 42–52,
face 53–66) and returns a ``HolisticPacket`` with wrist-anchor normalised
coordinates ready for ``GraphSequenceBuffer``.

The model asset is downloaded to ``~/.cache/maestro/`` on first use.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from maestro.domain.entities import FramePacket, HolisticPacket
from maestro.pipeline import graph_schema

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model asset
# ---------------------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)
_DEFAULT_CACHE = Path.home() / ".cache" / "maestro" / "holistic_landmarker.task"

# ---------------------------------------------------------------------------
# MediaPipe task-based API imports
# ---------------------------------------------------------------------------
_BaseOptions = mp.tasks.BaseOptions
_HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
_HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode


def _ensure_model(model_path: Path) -> None:
    """Download the HolisticLandmarker task bundle if not already cached."""
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _logger.info("Downloading holistic landmarker model to %s …", model_path)
    try:
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download holistic landmarker model from {_MODEL_URL}. "
            f"Provide the .task file manually via model_path= and try again. "
            f"Underlying error: {exc}"
        ) from exc
    _logger.info("Model downloaded successfully.")


class HolisticLandmarkExtractor:
    """Extracts 67 graph-schema-v1 keypoints using MediaPipe HolisticLandmarker.

    The extractor wraps the mediapipe 0.10+ task-based HolisticLandmarker and
    maps its outputs to the 67-node layout defined in ``graph_schema``:

      - Nodes   0–20: right hand (MediaPipe right-hand landmarks 0–20)
      - Nodes  21–41: left hand  (MediaPipe left-hand landmarks 0–20, offset +21)
      - Nodes  42–52: pose subset (11 selected landmarks per graph_schema.POSE_LANDMARK_INDICES)
      - Nodes  53–66: face subset (14 selected landmarks per graph_schema.FACE_LANDMARK_INDICES)

    Keypoints returned in ``HolisticPacket.keypoints`` are **always normalised**
    via ``graph_schema.normalise_graph_frame()`` so that the extractor and the
    dataset builder share identical coordinate spaces.

    Args:
        model_path: Path to the ``.task`` model bundle.  If ``None`` (default),
            the model is downloaded to ``~/.cache/maestro/`` on first use.
        model_complexity: 0 (lightweight) or 1 (accurate). Use 0 for real-time,
            1 for offline dataset construction.
        min_detection_confidence: Minimum detection confidence for hands/pose/face.
        min_tracking_confidence: Minimum landmark tracking confidence.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = Path(model_path) if model_path is not None else _DEFAULT_CACHE
        _ensure_model(resolved_path)

        # model_complexity is not directly exposed by the task API; we map it
        # to confidence thresholds — lower complexity equivalent is relaxed
        # thresholds for faster (less accurate) detection.
        if model_complexity == 0:
            min_detection_confidence = max(0.3, min_detection_confidence - 0.1)

        options = _HolisticLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=_RunningMode.VIDEO,
            min_face_detection_confidence=min_detection_confidence,
            min_face_landmarks_confidence=min_tracking_confidence,
            min_face_suppression_threshold=0.3,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_landmarks_confidence=min_tracking_confidence,
            min_pose_suppression_threshold=0.3,
            min_hand_landmarks_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )

        self._landmarker = _HolisticLandmarker.create_from_options(options)
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, frame: FramePacket) -> HolisticPacket:
        """Extract 67 normalised keypoints from *frame*.

        Args:
            frame: A ``FramePacket`` with a BGR ``frame.frame`` numpy array.

        Returns:
            ``HolisticPacket`` with wrist-normalised ``keypoints (67, 3)``
            and per-node ``valid_mask (67,)``.
        """
        rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Use the frame timestamp in milliseconds; must be monotonically increasing.
        result = self._landmarker.detect_for_video(mp_image, frame.timestamp_ms)
        self._frame_counter += 1

        keypoints, valid_mask = self._parse_result(result)

        # Apply canonical wrist-anchor normalisation (identical to dataset build path).
        keypoints = graph_schema.normalise_graph_frame(keypoints)

        return HolisticPacket(
            keypoints=keypoints.astype(np.float32),
            valid_mask=valid_mask,
            timestamp_ms=frame.timestamp_ms,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_result(
        self,
        result: mp.tasks.vision.HolisticLandmarkerResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map MediaPipe result to a ``(67, 3)`` keypoints array and bool mask.

        Undetected hands / face / pose nodes are zero-filled; the corresponding
        ``valid_mask`` positions are set to ``False``.
        """
        keypoints = np.zeros((graph_schema.NUM_NODES, graph_schema.NUM_COORDS), dtype=np.float32)
        valid_mask = np.zeros(graph_schema.NUM_NODES, dtype=bool)

        # --- Right hand (nodes 0–20) ---
        rh_lms = result.right_hand_landmarks
        if rh_lms:
            for i, lm in enumerate(rh_lms[:21]):
                if not _is_nan(lm):
                    keypoints[i] = [lm.x, lm.y, lm.z]
                    valid_mask[i] = True

        # --- Left hand (nodes 21–41) ---
        lh_lms = result.left_hand_landmarks
        lh_offset = graph_schema.NODE_INDEX_MAP["left_hand"][0]  # 21
        if lh_lms:
            for i, lm in enumerate(lh_lms[:21]):
                node_idx = i + lh_offset
                if not _is_nan(lm):
                    keypoints[node_idx] = [lm.x, lm.y, lm.z]
                    valid_mask[node_idx] = True

        # --- Pose subset (nodes 42–52) ---
        pose_lms = result.pose_landmarks
        if pose_lms:
            for graph_i, mp_pose_i in enumerate(graph_schema.POSE_LANDMARK_INDICES):
                node_idx = graph_schema.NODE_INDEX_MAP["pose"][0] + graph_i  # 42+
                if mp_pose_i < len(pose_lms):
                    lm = pose_lms[mp_pose_i]
                    if not _is_nan(lm):
                        keypoints[node_idx] = [lm.x, lm.y, lm.z]
                        valid_mask[node_idx] = True

        # --- Face subset (nodes 53–66) ---
        face_lms = result.face_landmarks
        if face_lms:
            for graph_i, mp_face_i in enumerate(graph_schema.FACE_LANDMARK_INDICES):
                node_idx = graph_schema.NODE_INDEX_MAP["face"][0] + graph_i  # 53+
                if mp_face_i < len(face_lms):
                    lm = face_lms[mp_face_i]
                    if not _is_nan(lm):
                        keypoints[node_idx] = [lm.x, lm.y, lm.z]
                        valid_mask[node_idx] = True

        return keypoints, valid_mask


def _is_nan(lm: object) -> bool:
    """Return True if any coordinate of *lm* is NaN."""
    return (
        math.isnan(getattr(lm, "x", 0.0))
        or math.isnan(getattr(lm, "y", 0.0))
        or math.isnan(getattr(lm, "z", 0.0))
    )
