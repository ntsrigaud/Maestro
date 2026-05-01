"""Graph schema — single source of truth for the 67-node skeleton.

All node indices, adjacency construction, and coordinate normalisation must
reference this module.  Never inline graph constants in other modules.

Schema version: graph-schema-v1
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Node index map
#
#   0 – 20   Right hand:  MediaPipe Hands landmarks 0–20 (selfie-view right)
#  21 – 41   Left hand:   MediaPipe Hands landmarks 0–20 (offset +21)
#  42 – 52   Pose subset: 11 selected MediaPipe Pose landmarks
#  53 – 66   Face subset: 14 MediaPipe Face Mesh contour keypoints
# ---------------------------------------------------------------------------

NODE_INDEX_MAP: dict[str, tuple[int, int]] = {
    "right_hand": (0, 20),
    "left_hand": (21, 41),
    "pose": (42, 52),
    "face": (53, 66),
}

NUM_NODES: int = 67
NUM_COORDS: int = 3
GRAPH_SCHEMA_VERSION: str = "graph-schema-v1"
WINDOW_LENGTH: int = 24
WINDOW_STEP: int = 3
MIN_VALID_RATIO: float = 0.8

# ---------------------------------------------------------------------------
# Pose landmark subset mapping (graph index → MediaPipe Pose landmark index)
#
#  Graph 42 → Pose  0  nose
#  Graph 43 → Pose  2  left eye inner
#  Graph 44 → Pose  5  right eye inner
#  Graph 45 → Pose  7  left ear
#  Graph 46 → Pose  8  right ear
#  Graph 47 → Pose 11  left shoulder
#  Graph 48 → Pose 12  right shoulder
#  Graph 49 → Pose 13  left elbow
#  Graph 50 → Pose 14  right elbow
#  Graph 51 → Pose 15  left wrist  (pose)
#  Graph 52 → Pose 16  right wrist (pose)
# ---------------------------------------------------------------------------
POSE_LANDMARK_INDICES: list[int] = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
_POSE_OFFSET: int = NODE_INDEX_MAP["pose"][0]  # 42

# ---------------------------------------------------------------------------
# Face landmark subset mapping (graph indices 53–66 → MediaPipe Face Mesh)
#
# 14 face contour keypoints selected for stability and expressiveness.
# Indices are from the 468-point MediaPipe Face Mesh topology.
# ---------------------------------------------------------------------------
FACE_LANDMARK_INDICES: list[int] = [
    10,  # forehead centre
    152,  # chin
    234,  # left cheek (face-right from model's POV)
    454,  # right cheek
    33,  # left eye outer corner
    263,  # right eye outer corner
    61,  # left mouth corner
    291,  # right mouth corner
    13,  # upper lip centre
    14,  # lower lip centre
    168,  # nose bridge
    1,  # nose tip
    57,  # left jaw
    287,  # right jaw
]
_FACE_OFFSET: int = NODE_INDEX_MAP["face"][0]  # 53

# ---------------------------------------------------------------------------
# MediaPipe hand skeleton — 20 edges per hand (standard fingertip chain)
# ---------------------------------------------------------------------------
_HAND_EDGES: list[tuple[int, int]] = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# ---------------------------------------------------------------------------
# Pose skeleton edges (within the 11-node pose subset, using graph indices)
# ---------------------------------------------------------------------------
_G = _POSE_OFFSET  # base offset = 42
_POSE_EDGES: list[tuple[int, int]] = [
    # Shoulders
    (_G + 5, _G + 6),  # left shoulder (47) — right shoulder (48)
    # Left arm
    (_G + 5, _G + 7),  # left shoulder (47) — left elbow (49)
    (_G + 7, _G + 9),  # left elbow (49) — left wrist (51)
    # Right arm
    (_G + 6, _G + 8),  # right shoulder (48) — right elbow (50)
    (_G + 8, _G + 10),  # right elbow (50) — right wrist (52)
    # Nose to shoulders
    (_G + 0, _G + 5),  # nose (42) — left shoulder (47)
    (_G + 0, _G + 6),  # nose (42) — right shoulder (48)
    # Eyes / ears
    (_G + 0, _G + 1),  # nose (42) — left eye inner (43)
    (_G + 0, _G + 2),  # nose (42) — right eye inner (44)
    (_G + 1, _G + 3),  # left eye inner (43) — left ear (45)
    (_G + 2, _G + 4),  # right eye inner (44) — right ear (46)
]

# ---------------------------------------------------------------------------
# Cross-wrist bridges: hand wrists ↔ pose wrists
#   Right hand wrist: graph node 0  → right pose wrist: graph node 52
#   Left hand wrist:  graph node 21 → left pose wrist:  graph node 51
# ---------------------------------------------------------------------------
_CROSS_EDGES: list[tuple[int, int]] = [
    (0, 52),  # right hand wrist → right pose wrist
    (21, 51),  # left hand wrist  → left pose wrist
]

# ---------------------------------------------------------------------------
# Face skeleton edges (minimal face contour ring)
# ---------------------------------------------------------------------------
_F = _FACE_OFFSET  # base offset = 53
_FACE_EDGES: list[tuple[int, int]] = [
    # Forehead → cheeks
    (_F + 0, _F + 2),  # forehead (53) — left cheek (55)
    (_F + 0, _F + 3),  # forehead (53) — right cheek (56)
    # Cheeks → chin
    (_F + 2, _F + 1),  # left cheek (55) — chin (54)
    (_F + 3, _F + 1),  # right cheek (56) — chin (54)
    # Eye corners
    (_F + 4, _F + 5),  # left eye outer (57) — right eye outer (58)
    # Mouth
    (_F + 6, _F + 7),  # left mouth corner (59) — right mouth corner (60)
    (_F + 8, _F + 9),  # upper lip (61) — lower lip (62)
    # Nose
    (_F + 10, _F + 11),  # nose bridge (63) — nose tip (64)
    # Jaw
    (_F + 12, _F + 13),  # left jaw (65) — right jaw (66)
    # Connect face to pose nose
    (_G + 0, _F + 11),  # pose nose (42) — face nose tip (64)
]


def build_adjacency() -> np.ndarray:
    """Build the binary symmetric adjacency matrix of shape ``(67, 67)``.

    Includes edges for:
    - Both hand skeletons (20 edges each)
    - Cross-wrist bridges connecting hand wrists to pose wrists
    - Pose arm and head skeleton
    - Face contour ring
    - Face-pose connection at the nose

    Returns:
        ``np.ndarray`` of shape ``(67, 67)``, dtype ``float32``, zero diagonal.
    """
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    # Right hand edges (base indices as-is)
    for u, v in _HAND_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Left hand edges (offset by 21)
    lh_offset = NODE_INDEX_MAP["left_hand"][0]  # 21
    for u, v in _HAND_EDGES:
        A[u + lh_offset, v + lh_offset] = 1.0
        A[v + lh_offset, u + lh_offset] = 1.0

    # Pose edges
    for u, v in _POSE_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Cross-wrist bridges
    for u, v in _CROSS_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Face edges
    for u, v in _FACE_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Ensure zero diagonal (self-loops excluded)
    np.fill_diagonal(A, 0.0)
    return A


def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Return the degree-normalised adjacency: ``D^{-1/2} (A + I) D^{-1/2}``.

    Adds self-loops before normalisation so each node attends to itself.

    Args:
        A: Binary adjacency matrix ``(N, N)``.

    Returns:
        Normalised adjacency ``np.ndarray`` of the same shape, ``float32``.
    """
    A = A.astype(np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)  # add self-loops
    degree = A_hat.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0.0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def normalise_graph_frame(frame: np.ndarray) -> np.ndarray:
    """Apply wrist-anchor normalisation to a single raw graph frame.

    This normalisation is applied **identically** during dataset construction
    and at inference time so that training and inference coordinates occupy the
    same canonical space.

    Procedure:
      1. Compute anchor as the mean of both wrist positions (nodes 0 and 21).
         If a hand is absent (zero-filled) use only the visible wrist.
      2. Subtract the anchor from all 67 nodes.
      3. Compute scale as the Euclidean distance between the two wrists
         (||node_0 – node_21||₂).  Fall back to the mean inter-knuckle span
         if both wrists are absent or the distance is degenerate (< 0.01).
      4. Divide all coordinates by scale.

    Args:
        frame: ``np.ndarray`` of shape ``(67, 3)``, raw MediaPipe normalised
               coordinates in range roughly ``[0, 1]``.

    Returns:
        ``np.ndarray`` of shape ``(67, 3)``, ``float32``, wrist-normalised.
    """
    frame = frame.astype(np.float32)

    rh_wrist = frame[0]  # right hand wrist
    lh_wrist = frame[21]  # left hand wrist

    rh_present = np.any(rh_wrist != 0.0)
    lh_present = np.any(lh_wrist != 0.0)

    if rh_present and lh_present:
        anchor = (rh_wrist + lh_wrist) / 2.0
    elif rh_present:
        anchor = rh_wrist.copy()
    elif lh_present:
        anchor = lh_wrist.copy()
    else:
        # No hand detected; centre on mean of all non-zero nodes.
        nonzero_mask = np.any(frame != 0.0, axis=1)
        if nonzero_mask.any():
            anchor = frame[nonzero_mask].mean(axis=0)
        else:
            anchor = np.zeros(NUM_COORDS, dtype=np.float32)

    frame = frame - anchor

    # Scale by inter-wrist distance when both hands are present.
    if rh_present and lh_present:
        scale = float(np.linalg.norm(frame[0] - frame[21]))
    else:
        scale = 0.0

    # Fallback: mean span across detected hand knuckle pairs.
    if scale < 0.01:
        # Use distance between MCP joints of index and pinky fingers as proxy.
        # Right hand: nodes 5 (index MCP) and 17 (pinky MCP)
        # Left hand: nodes 26 (index MCP, offset) and 38 (pinky MCP, offset)
        spans = []
        if rh_present:
            spans.append(float(np.linalg.norm(frame[5] - frame[17])))
        if lh_present:
            spans.append(float(np.linalg.norm(frame[26] - frame[38])))
        if spans:
            scale = float(np.mean(spans))

    if scale < 1e-6:
        scale = 1.0  # last resort: avoid division by zero

    frame = frame / scale
    return frame
