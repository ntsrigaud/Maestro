from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TwoStreamGestureLSTM(nn.Module):
    """Two-stream LSTM classifier with per-stream attention pooling.

    The 147-dim feature-schema-v5 input is partitioned into two semantically
    coherent streams before any recurrent processing occurs:

    Stream A — Pose / Shape (73 dims)
        dims  0:63   positions       (21 wrist-relative landmarks × 3)
        dims 134:144 joint_angles    (10 MCP + PIP joint angles)

    Stream B — Motion / Dynamics (74 dims)
        dims  63:68  fingertip_spread (5 inter-fingertip distances)
        dims  68:71  wrist_trajectory (3 net wrist displacement)
        dims  71:134 velocity         (21 per-landmark velocity vectors × 3)
        dims 144:147 wrist_vel_raw    (3 camera-normalised wrist velocity)

    Each stream runs a 2-layer LSTM with additive (Bahdanau-style) temporal
    attention, producing a context vector of shape (B, hidden_size).  The two
    context vectors are concatenated (B, 2×hidden_size) and passed through a
    Dropout + Linear classification head.

    This architecture explicitly gives motion features a dedicated
    representational pathway, addressing the signal-dilution problem identified
    in the model-improvement-plan (Tier 4-A): the 74 motion dims were receiving
    the same gradient weight as the 73 pose dims inside the single-stream LSTM,
    even though pose dims vastly outnumber the 6 critical directional dims.

    Parameter budget (hidden_size=64 per stream):
        LSTM-A:  4 × 64 × (73 + 64 + 1)  × 2 layers  ≈  90 K
        LSTM-B:  4 × 64 × (74 + 64 + 1)  × 2 layers  ≈  91 K
        Head  :  128 × 8 + 8              ≈   1 K
        Attn  :  64 + 64                  ≈   0 K
        Total                             ≈ 182 K  (vs ~264 K for single-stream)
    """

    # Feature index splits (exclusive end indices, Python slice notation)
    _POSE_SLICES: tuple[tuple[int, int], ...] = ((0, 63), (134, 144))
    _MOTION_SLICES: tuple[tuple[int, int], ...] = ((63, 134), (144, 147))

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        """Initialise the two-stream LSTM.

        Args:
            input_size: Total feature dimensionality.  Must equal 147 for
                feature-schema-v5; stored for artifact serialisation only.
            hidden_size: Hidden units *per stream*.  The fusion layer receives
                ``2 × hidden_size`` units, so the effective classification
                bottleneck is twice this value.  Recommended: 64.
            num_layers: LSTM depth for both streams.  Recommended: 2.
            dropout: Applied both inside the multi-layer LSTMs (between layers)
                and in the pre-head Dropout.
            num_classes: Output class count (vocabulary size).
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        _lstm_kwargs = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Stream A: pose + shape features (73 dims)
        self.lstm_a = nn.LSTM(input_size=73, **_lstm_kwargs)  # type: ignore[arg-type]
        self.attn_a = nn.Linear(hidden_size, 1)

        # Stream B: motion + dynamics features (74 dims)
        self.lstm_b = nn.LSTM(input_size=74, **_lstm_kwargs)  # type: ignore[arg-type]
        self.attn_b = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(2 * hidden_size, num_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partition a (B, T, 147) tensor into pose (B, T, 73) and motion (B, T, 74).

        Pose   = positions[0:63]   + joint_angles[134:144]   → 73 dims
        Motion = fingertip[63:68]  + wrist_traj[68:71]
               + velocity[71:134]  + wrist_vel[144:147]      → 74 dims
        """
        pose = torch.cat([x[..., :63], x[..., 134:144]], dim=-1)  # (B, T, 73)
        motion = torch.cat([x[..., 63:134], x[..., 144:]], dim=-1)  # (B, T, 74)
        return pose, motion

    @staticmethod
    def _attend(output: torch.Tensor, attn: nn.Linear) -> torch.Tensor:
        """Compute attention-weighted context vector over time.

        Args:
            output: LSTM output (B, T, H).
            attn: Attention scoring layer Linear(H, 1).

        Returns:
            Context vector (B, H).
        """
        scores = attn(output)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        return (weights * output).sum(dim=1)  # (B, H)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, input_size) where T is the
               sequence length (typically 24) and input_size is 147.

        Returns:
            Logits tensor of shape (B, num_classes).  Apply softmax externally
            for probabilities (inference) or pass directly to
            ``nn.CrossEntropyLoss`` during training.
        """
        pose, motion = self._split(x)

        out_a, _ = self.lstm_a(pose)  # (B, T, H)
        ctx_a = self._attend(out_a, self.attn_a)  # (B, H)

        out_b, _ = self.lstm_b(motion)  # (B, T, H)
        ctx_b = self._attend(out_b, self.attn_b)  # (B, H)

        fused = torch.cat([ctx_a, ctx_b], dim=-1)  # (B, 2H)
        return self.head(self.dropout(fused))  # (B, C)
