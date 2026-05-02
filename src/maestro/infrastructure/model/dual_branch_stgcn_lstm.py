"""Dual-branch STGCN-LSTM gesture classifier (phonological + skeleton branches).

Implements the architecture from:
  "A Study on the STGCN-LSTM Sign Language Recognition Model Based on
   Phonological Features of Sign Language" (IEEE, 2025)
   https://ieeexplore.ieee.org/document/10965662

Adapted for the IPN Hand skeleton dataset (67-node graph-schema-v1 input).

Architecture overview
─────────────────────
Input: (B, T, N=67, C=3) graph-schema-v1 normalised skeleton sequence.

Branch 1 — Phonological SA-ConvLSTM
  Implements the paper's λ(k) phonological feature extraction (Eq. 1–5):

    λ(k) = [H(k), n(k), l(k), μ(k)]

  where, per frame k:
    H(k) — hand shape:  15 finger-joint angles β_ij (angle at each PIP/DIP/
            IP joint, both hands) + 4 inter-finger angles ω_i (angle between
            adjacent proximal phalanges, both hands) → 38 values total.
    n(k) — palm direction: unit normal of wrist→index-MCP, wrist→pinky-MCP
            triangle, both hands → 6 values.
    l(k) — hand position: middle-finger MCP coordinate, both hands → 6 values.
    μ(k) — motion: velocity v_k, acceleration a_k, curvature s_k of the
            middle-MCP trajectory, both hands → 6 values per frame.

  The feature vector is then passed through an SA-ConvLSTM pipeline (Eq. 6–8):
    λ(k)                                  (B, T, phono_feat_dim)
    → Temporal Conv1d + BN + ReLU         (B, T, conv_ch)
    → Stacked LSTM                        (B, T, lstm_hidden)
    → Full Q/K/V self-attention + residual (B, T, lstm_hidden)  ← Eq. 6-8
    → Bahdanau temporal attention          (B, lstm_hidden)

  Note: hand-shape in the original paper uses SVM top-3 labels.  Here we use
  the raw geometric angles directly (end-to-end differentiable, appropriate for
  a neural pipeline without a separate SVM stage).

Branch 2 — STGCN (3 blocks, fully-connected graph)
  Three STGCNBlock layers operating on a K=3 partitioned fully-connected
  adjacency (all nodes attend to all nodes; centripetal / centrifugal split
  derived from anatomical BFS distances).

  Each block follows the paper's SCN→TCN→BN→ReLU→Dropout→+Residual order:
    (B, T, N, C)
      → STGCNBlock(C   → 64,  K=3, residual=True, dropout=p)
      → STGCNBlock(64  → 128, K=3, residual=True, dropout=p)
      → STGCNBlock(128 → 256, K=3, residual=True, dropout=p)
      → mean-pool over N                     (B, T, 256)
      → mean-pool over T                     (B, 256)

Fusion
  cat(branch1, branch2) → Dropout → Linear → (B, num_classes)

Phonological feature dimension (phono_feat_dim)
  Per hand: 15 joint angles + 4 inter-finger + 3 palm normal + 3 position
           + 3 motion = 28; both hands → 56.  Total phono_feat_dim = 56.

References
──────────
  Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition", AAAI 2018.
  Shi et al., "Skeleton-Based Action Recognition with Directed Graph Neural
  Networks", CVPR 2019.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from maestro.infrastructure.model.stgcn_lstm import (
    STGCNBlock,
    _BahdanauAttention,
)

# ─────────────────────────────────────────────────────────────────────────────
# Node index constants (graph-schema-v1)
# Right hand: nodes 0-20; Left hand: nodes 21-41 (offset +21)
# ─────────────────────────────────────────────────────────────────────────────

# Right hand joint indices (MediaPipe layout)
_R_WRIST = 0
_R_THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
_R_INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
_R_MIDDLE = (9, 10, 11, 12)
_R_RING = (13, 14, 15, 16)
_R_PINKY = (17, 18, 19, 20)

# Left hand: same layout, offset +21
_L_OFFSET = 21
_L_WRIST = _L_OFFSET + 0
_L_THUMB = tuple(i + _L_OFFSET for i in _R_THUMB)
_L_INDEX = tuple(i + _L_OFFSET for i in _R_INDEX)
_L_MIDDLE = tuple(i + _L_OFFSET for i in _R_MIDDLE)
_L_RING = tuple(i + _L_OFFSET for i in _R_RING)
_L_PINKY = tuple(i + _L_OFFSET for i in _R_PINKY)

# Finger joint angle triples: (prev_joint, pivot, next_joint) — one angle per
# non-endpoint joint.  5 fingers × 3 intermediate joints = 15 per hand.
# Each tuple: (parent_node, pivot_node, child_node).
_JOINT_ANGLE_TRIPLES_RIGHT: list[tuple[int, int, int]] = [
    # Thumb: CMC(1), MCP(2), IP(3)
    (_R_WRIST, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index: wrist(0), MCP(5), PIP(6), DIP(7)
    (_R_WRIST, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle
    (_R_WRIST, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring
    (_R_WRIST, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky
    (_R_WRIST, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]
_JOINT_ANGLE_TRIPLES_LEFT: list[tuple[int, int, int]] = [
    (a + _L_OFFSET, b + _L_OFFSET, c + _L_OFFSET) for a, b, c in _JOINT_ANGLE_TRIPLES_RIGHT
]

# Inter-finger angle pairs: vectors from wrist to MCP of each finger;
# 4 adjacent pairs per hand (thumb MCP=2, index MCP=5, middle=9, ring=13, pinky=17).
_INTER_FINGER_MCP_RIGHT = [2, 5, 9, 13, 17]  # 5 entries → 4 pairs
_INTER_FINGER_MCP_LEFT = [i + _L_OFFSET for i in _INTER_FINGER_MCP_RIGHT]

# Middle-finger MCP (landmark for hand position / motion)
_R_MIDDLE_MCP = 9
_L_MIDDLE_MCP = _L_OFFSET + 9

# Index MCP and Pinky MCP for palm normal
_R_INDEX_MCP = 5
_R_PINKY_MCP = 17
_L_INDEX_MCP = _L_OFFSET + 5
_L_PINKY_MCP = _L_OFFSET + 17

# Phonological feature dimension per hand: 15 joint + 4 inter-finger + 3 normal
# + 3 position + 3 motion = 28.  Two hands → 56.
PHONO_FEAT_DIM: int = 56


def _safe_angle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Angle between two vectors, numerically safe. Returns radians in [0, π].

    Args:
        v1: (..., 3)
        v2: (..., 3)

    Returns:
        (...,) angle in radians.
    """
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
    return torch.arccos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def _extract_hand_features(
    x: torch.Tensor,
    wrist_idx: int,
    joint_triples: list[tuple[int, int, int]],
    inter_mcp: list[int],
    index_mcp: int,
    pinky_mcp: int,
    middle_mcp: int,
) -> torch.Tensor:
    """Compute phonological feature vector for one hand per frame.

    Args:
        x:             Skeleton tensor ``(B, T, N, 3)``.
        wrist_idx:     Node index of the wrist.
        joint_triples: List of (parent, pivot, child) index triples for
                       joint-angle computation (15 entries).
        inter_mcp:     Node indices of the 4 finger MCP joints for
                       inter-finger angle computation.
        index_mcp:     Node index of index-finger MCP (for palm normal).
        pinky_mcp:     Node index of pinky MCP (for palm normal).
        middle_mcp:    Node index of middle-finger MCP (position / motion).

    Returns:
        ``(B, T, 28)`` — per-frame phonological features for this hand.
    """
    B, T, N, _ = x.shape

    # ── 15 joint angles ──────────────────────────────────────────────────────
    joint_angles: list[torch.Tensor] = []
    for a, b, c in joint_triples:
        pa = x[:, :, a, :]  # (B, T, 3) — parent
        pb = x[:, :, b, :]  # pivot
        pc = x[:, :, c, :]  # child
        v1 = pa - pb  # vector toward parent
        v2 = pc - pb  # vector toward child
        joint_angles.append(_safe_angle(v1, v2).unsqueeze(-1))  # (B,T,1)
    h_shape = torch.cat(joint_angles, dim=-1)  # (B, T, 15)

    # ── 4 inter-finger angles (adjacent MCP vectors from wrist) ──────────────
    wrist = x[:, :, wrist_idx, :]  # (B, T, 3)
    inter_angles: list[torch.Tensor] = []
    for i in range(len(inter_mcp) - 1):
        v1 = x[:, :, inter_mcp[i], :] - wrist
        v2 = x[:, :, inter_mcp[i + 1], :] - wrist
        inter_angles.append(_safe_angle(v1, v2).unsqueeze(-1))  # (B,T,1)
    h_inter = torch.cat(inter_angles, dim=-1)  # (B, T, 4)

    # ── Palm normal: cross(wrist→index_MCP, wrist→pinky_MCP) ─────────────────
    v_index = x[:, :, index_mcp, :] - wrist  # (B, T, 3)
    v_pinky = x[:, :, pinky_mcp, :] - wrist
    normal = torch.linalg.cross(v_index, v_pinky, dim=-1)  # (B, T, 3)
    norm_mag = normal.norm(dim=-1, keepdim=True) + 1e-8
    normal = normal / norm_mag  # unit normal

    # ── Hand position: middle-finger MCP coordinates ──────────────────────────
    position = x[:, :, middle_mcp, :]  # (B, T, 3)

    # ── Motion: velocity, acceleration, curvature of middle-MCP trajectory ────
    # Velocity: frame-to-frame Euclidean distance  (B, T, 1) — pad frame 0
    diff = position[:, 1:, :] - position[:, :-1, :]  # (B, T-1, 3)
    vel = diff.norm(dim=-1)  # (B, T-1)
    vel = torch.cat([vel[:, :1], vel], dim=1)  # (B, T)

    # Acceleration: Δvelocity, pad frame 0
    acc = vel[:, 1:] - vel[:, :-1]  # (B, T-1)
    acc = torch.cat([acc[:, :1], acc], dim=1)  # (B, T)

    # Curvature: (d_{k-1,k} + d_{k,k+1}) / d_{k-1,k+1}  ─ pad boundaries
    d_prev = torch.cat([vel[:, :1], vel[:, :-1]], dim=1)  # shift right
    d_next = torch.cat([vel[:, 1:], vel[:, -1:]], dim=1)  # shift left
    p_prev = torch.cat([position[:, :1], position[:, :-1]], dim=1)
    p_next = torch.cat([position[:, 1:], position[:, -1:]], dim=1)
    chord = (p_next - p_prev).norm(dim=-1) + 1e-8  # (B, T)
    curv = (d_prev + d_next) / chord  # (B, T)

    motion = torch.stack([vel, acc, curv], dim=-1)  # (B, T, 3)

    return torch.cat([h_shape, h_inter, normal, position, motion], dim=-1)  # (B,T,28)


class _PhonologicalFeatureExtractor(nn.Module):
    """Extract structured phonological features from a raw skeleton tensor.

    Computes the λ(k) feature vector per frame as described in the paper:
      λ(k) = [H(k), n(k), l(k), μ(k)]

    Both right and left hands are processed; their 28-dim feature vectors are
    concatenated to give a 56-dim output per frame.

    This module has **no learnable parameters** — it is a deterministic
    geometric transform.

    Args:
        None  (operates on fixed graph-schema-v1 node layout).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, PHONO_FEAT_DIM=56)`` phonological feature tensor."""
        r = _extract_hand_features(
            x,
            wrist_idx=_R_WRIST,
            joint_triples=_JOINT_ANGLE_TRIPLES_RIGHT,
            inter_mcp=_INTER_FINGER_MCP_RIGHT,
            index_mcp=_R_INDEX_MCP,
            pinky_mcp=_R_PINKY_MCP,
            middle_mcp=_R_MIDDLE_MCP,
        )  # (B, T, 28)
        left = _extract_hand_features(
            x,
            wrist_idx=_L_WRIST,
            joint_triples=_JOINT_ANGLE_TRIPLES_LEFT,
            inter_mcp=_INTER_FINGER_MCP_LEFT,
            index_mcp=_L_INDEX_MCP,
            pinky_mcp=_L_PINKY_MCP,
            middle_mcp=_L_MIDDLE_MCP,
        )  # (B, T, 28)
        return torch.cat([r, left], dim=-1)  # (B, T, 56)


# ─────────────────────────────────────────────────────────────────────────────
# Full Q/K/V self-attention (SA block, Eq. 6–8 in the paper)
# ─────────────────────────────────────────────────────────────────────────────


class _SelfAttentionQKV(nn.Module):
    """Full dot-product Q/K/V self-attention over the sequence dimension.

    Implements the SA-ConvLSTM attention equations from the paper (Eq. 6–8):

        Q = W_q · H_t,   K = W_k · H_t,   V = W_v · H_t
        e_{i,j}   = Q_i · K_j^T            (scaled by 1/√D)
        α_{i,j}   = softmax_j(e_{i,j})
        Z_i       = Σ_j α_{i,j} V_j
        Ĥ_t       = w_f · Z + H_t          (learnable residual gate)

    Applied over the time dimension T of an (B, T, D) tensor.

    Args:
        dim: Feature dimension D.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        # Learnable scalar residual gate (initialised to 1)
        self.w_f = nn.Parameter(torch.ones(1))
        self.scale = dim**-0.5

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and residual.

        Args:
            H: ``(B, T, D)``

        Returns:
            ``(B, T, D)`` — attention output with residual.
        """
        Q = self.W_q(H)  # (B, T, D)
        K = self.W_k(H)
        V = self.W_v(H)
        # e_{i,j} = Q_i · K_j^T  →  (B, T, T)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        alpha = F.softmax(attn, dim=-1)  # (B, T, T)
        Z = torch.bmm(alpha, V)  # (B, T, D)
        return self.w_f * Z + H  # residual (Eq. 8)


# ─────────────────────────────────────────────────────────────────────────────
# Phonological branch
# ─────────────────────────────────────────────────────────────────────────────


class _PhonologicalBranch(nn.Module):
    """Skeleton-adapted SA-ConvLSTM branch.

    Implements Branch 1 of the dual-branch model following the paper:

    1. Extract λ(k) phonological features per frame (geometric, parameter-free).
    2. Temporal Conv1d + BN + ReLU — local temporal pattern capture.
    3. Stacked LSTM — long-range temporal modelling.
    4. Full Q/K/V self-attention + residual (SA block, Eq. 6–8).
    5. Bahdanau attention → context vector.

    Args:
        conv_channels: Output channels of the temporal convolution.
        lstm_hidden:   Hidden size of the LSTM.
        lstm_layers:   Number of stacked LSTM layers.
        dropout:       Dropout between LSTM layers (only applied when ≥2 layers).
    """

    def __init__(
        self,
        conv_channels: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Step 1: deterministic phonological feature extraction
        self.feature_extractor = _PhonologicalFeatureExtractor()

        # Step 2: temporal 1-D convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(PHONO_FEAT_DIM, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # Step 3: LSTM for long-range temporal modelling
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Step 4: Full Q/K/V self-attention (SA block, Eq. 6–8)
        self.self_attn = _SelfAttentionQKV(lstm_hidden)

        # Step 5: Bahdanau temporal attention → context vector
        self.temporal_attn = _BahdanauAttention(lstm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return phonological feature vector ``(B, lstm_hidden)``.

        Args:
            x: ``(B, T, N, C)`` raw skeleton tensor.
        """
        # Compute λ(k) phonological features
        h = self.feature_extractor(x)  # (B, T, 56)

        # Temporal conv: needs (B, C, T)
        h = self.temporal_conv(h.permute(0, 2, 1))  # (B, conv_ch, T)
        h = h.permute(0, 2, 1)  # (B, T, conv_ch)

        # LSTM
        lstm_out, _ = self.lstm(h)  # (B, T, lstm_hidden)

        # Full Q/K/V self-attention with residual (SA block)
        sa_out = self.self_attn(lstm_out)  # (B, T, lstm_hidden)

        # Bahdanau temporal attention → (B, lstm_hidden)
        return self.temporal_attn(sa_out)


# ─────────────────────────────────────────────────────────────────────────────
# Main dual-branch model
# ─────────────────────────────────────────────────────────────────────────────


class DualBranchSTGCN_LSTM(nn.Module):
    """Dual-branch STGCN-LSTM gesture classifier.

    Implements the two-branch fusion architecture from the reference paper:

    * **Branch 1** (phonological / SA-ConvLSTM): computes structured geometric
      phonological features λ(k) = [H(k), n(k), l(k), μ(k)] per frame —
      finger-joint angles, inter-finger angles, palm normal, hand position, and
      motion — then processes them with Conv1d → LSTM → full Q/K/V SA block
      → Bahdanau attention.
    * **Branch 2** (STGCN, 3 blocks): captures spatio-temporal dependencies
      in the skeleton graph using three stacked STGCNBlocks operating on a
      fully-connected K=3 partitioned adjacency.

    The two branches are concatenated and classified via a linear head.

    Args:
        num_nodes:         Number of graph nodes (default 67; must be 67 for
                           the hardcoded IPN Hand node layout).
        num_coords:        Coordinate channels per node (default 3).
        phono_conv_channels: Temporal conv channels for Branch 1 (default 256).
        phono_lstm_hidden: LSTM hidden size for Branch 1 (default 256).
        phono_lstm_layers: LSTM layers for Branch 1 (default 2).
        stgcn_channels:    Channel widths for the 3 STGCN blocks (default
                           (64, 128, 256)).
        dropout:           Dropout before classification head (default 0.3).
        num_classes:       Number of gesture classes (default 8).
        adjacency:         Pre-computed adjacency tensor ``(K, N, N)`` for
                           the STGCN branch.  If ``None``, the fully-connected
                           K=3 partitioned adjacency is built automatically.
    """

    def __init__(
        self,
        *,
        num_nodes: int = 67,
        num_coords: int = 3,
        phono_conv_channels: int = 256,
        phono_lstm_hidden: int = 256,
        phono_lstm_layers: int = 2,
        stgcn_channels: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.3,
        num_classes: int = 8,
        adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.num_coords = num_coords
        self.phono_conv_channels = phono_conv_channels
        self.phono_lstm_hidden = phono_lstm_hidden
        self.phono_lstm_layers = phono_lstm_layers
        self.stgcn_channels = stgcn_channels
        self.dropout_p = dropout
        self.num_classes = num_classes

        # ── Build adjacency for STGCN branch ──────────────────────────────
        if adjacency is None:
            from maestro.pipeline.graph_schema import (
                build_fully_connected_partitioned_adjacency,
                normalise_partitioned_adjacency,
            )

            A = torch.from_numpy(
                normalise_partitioned_adjacency(
                    build_fully_connected_partitioned_adjacency()
                ).astype("float32")
            )  # (3, N, N)
        else:
            A = adjacency.float() if adjacency.dim() == 3 else adjacency.float().unsqueeze(0)

        self.register_buffer("adjacency", A)

        # ── Branch 1: Phonological SA-ConvLSTM ────────────────────────────
        self.phono_branch = _PhonologicalBranch(
            conv_channels=phono_conv_channels,
            lstm_hidden=phono_lstm_hidden,
            lstm_layers=phono_lstm_layers,
            dropout=dropout,
        )

        # ── Branch 2: STGCN (3 blocks, fully-connected graph) ─────────────
        stgcn_blocks: list[nn.Module] = []
        in_ch = num_coords
        for i in range(len(stgcn_channels)):
            out_ch = stgcn_channels[i]
            stgcn_blocks.append(
                STGCNBlock(
                    in_ch,
                    out_ch,
                    A,
                    temporal_kernel=3,
                    residual=True,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.stgcn_blocks = nn.ModuleList(stgcn_blocks)

        stgcn_out_dim = stgcn_channels[-1]

        # ── Fusion head ───────────────────────────────────────────────────
        fusion_dim = phono_lstm_hidden + stgcn_out_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(fusion_dim, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of gesture windows.

        Args:
            x: Float tensor ``(B, T, N, C)`` — graph-schema-v1 skeleton.

        Returns:
            Logits ``(B, num_classes)``.
        """
        # Branch 1: phonological SA-ConvLSTM → (B, phono_lstm_hidden)
        phono_feat = self.phono_branch(x)

        # Branch 2: 3 × STGCN → global pool → (B, stgcn_out_dim)
        h = x
        for block in self.stgcn_blocks:
            h = block(h)  # (B, T', N, C_out)
        stgcn_feat = h.mean(dim=(1, 2))  # mean over T and N → (B, C_out)

        # Fusion: concatenate, dropout, classify
        fused = torch.cat([phono_feat, stgcn_feat], dim=-1)  # (B, fusion_dim)
        return self.head(self.dropout(fused))  # (B, num_classes)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
