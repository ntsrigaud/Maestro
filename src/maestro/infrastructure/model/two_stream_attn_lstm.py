from __future__ import annotations

import torch
from torch import nn


class EnhancedTwoStreamLSTM(nn.Module):
    """Improved two-stream gesture classifier: BiLSTM + Multi-Head Self-Attention.

    Key architectural differences from :class:`TwoStreamGestureLSTM`:

    1. **Feature projection** — both streams (73 dims and 74 dims) are projected
       to a shared ``proj_dim`` before LSTM processing, making the two BiLSTMs
       fully symmetric and decoupling raw feature width from model capacity.

    2. **Bidirectional LSTMs** — forward *and* backward temporal context are
       captured simultaneously over the fixed-length input window.  Because
       gesture classification operates over a fully-buffered window (not a causal
       stream), bidirectionality is appropriate and consistently improves accuracy
       on fixed-window skeleton datasets.

    3. **Multi-Head Self-Attention (Q/K/V)** — replaces the single-head Bahdanau
       additive attention (``Linear(H, 1)``).  Each of the T output frames attends
       to all other frames across ``num_heads`` independent subspaces, yielding a
       richer temporal summary before mean-pooling.  A pre-norm residual structure
       (LayerNorm → MHA → residual + LayerNorm) is used for training stability.

    4. **Cross-stream gating** — after temporal pooling, each context vector is
       element-wise scaled by a sigmoid gate conditioned on the *other* stream's
       context.  This allows the motion stream to suppress or amplify pose
       representations when the two streams give conflicting signals (e.g. a fast
       swipe with a static hand shape).

    5. **Deeper MLP head** — ``LayerNorm → Linear → GELU → Dropout → Linear``
       replaces the single ``Linear(2H, C)`` layer.

    Feature partition (feature-schema-v5, 147 dims total):
        Stream A — Pose / Shape (73 dims)
            dims   0:63   positions       (21 wrist-relative landmarks × 3)
            dims 134:144  joint_angles    (10 MCP + PIP joint angles)

        Stream B — Motion / Dynamics (74 dims)
            dims  63:68   fingertip_spread (5 inter-fingertip distances)
            dims  68:71   wrist_trajectory (3 net wrist displacement)
            dims  71:134  velocity         (21 per-landmark velocity × 3)
            dims 144:147  wrist_vel_raw    (3 camera-normalised wrist velocity)

    Parameter budget (proj_dim=64, hidden_size=64, num_layers=2, num_heads=4):
        proj_a / proj_b         2 × ~5 K              ≈  10 K
        BiLSTM-A                layer1 66 K + layer2 99 K  ≈ 165 K
        BiLSTM-B                (identical after projection)  ≈ 165 K
        LayerNorm × 4           negligible             ≈   1 K
        MHA-A / MHA-B           2 × (4 × 128² + bias) ≈ 132 K
        Cross-gates × 2         2 × 128²               ≈  33 K
        Head MLP                256 → 128 → C          ≈  35 K
        Total                                          ≈ 541 K
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        num_heads: int = 4,
        proj_dim: int = 64,
    ) -> None:
        """Initialise the enhanced two-stream LSTM.

        Args:
            input_size: Total feature dimensionality (must be 147 for
                feature-schema-v5); stored for artifact serialisation only.
            hidden_size: Hidden units *per direction per stream*.  BiLSTM output
                width = ``2 × hidden_size``.  Recommended: 64.
            num_layers: LSTM depth.  Dropout is applied between layers when
                ``num_layers > 1``.  Recommended: 2.
            dropout: Dropout rate applied between LSTM layers, inside MHA, and
                in the pre-head Dropout.
            num_classes: Output vocabulary size.
            num_heads: Number of attention heads.  Must divide
                ``2 × hidden_size`` evenly.  Default: 4.
            proj_dim: Output dimension of the per-stream feature projection.
                Recommended: same as ``hidden_size`` (64).
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.proj_dim = proj_dim

        bilstm_out = hidden_size * 2  # bidirectional doubles hidden width

        if bilstm_out % num_heads != 0:
            raise ValueError(
                f"BiLSTM output width ({bilstm_out} = 2 × hidden_size={hidden_size}) "
                f"must be divisible by num_heads={num_heads}."
            )

        # ── Feature projection ────────────────────────────────────────────────
        self.proj_a = nn.Sequential(
            nn.Linear(73, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_b = nn.Sequential(
            nn.Linear(74, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # ── Bidirectional LSTMs ───────────────────────────────────────────────
        _lstm_kw: dict = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.lstm_a = nn.LSTM(input_size=proj_dim, **_lstm_kw)
        self.lstm_b = nn.LSTM(input_size=proj_dim, **_lstm_kw)

        # ── Pre-attention LayerNorm ───────────────────────────────────────────
        self.ln_a = nn.LayerNorm(bilstm_out)
        self.ln_b = nn.LayerNorm(bilstm_out)

        # ── Multi-Head Self-Attention (Q/K/V) ─────────────────────────────────
        # batch_first=True → input / output shapes are (B, T, E).
        _mha_kw: dict = dict(
            embed_dim=bilstm_out,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mha_a = nn.MultiheadAttention(**_mha_kw)
        self.mha_b = nn.MultiheadAttention(**_mha_kw)

        # Post-attention LayerNorm (applied to residual-fused output)
        self.ln_post_a = nn.LayerNorm(bilstm_out)
        self.ln_post_b = nn.LayerNorm(bilstm_out)

        # ── Cross-stream gating ───────────────────────────────────────────────
        # gate_a: σ(Linear(ctx_b)) → element-wise scale for ctx_a
        # gate_b: σ(Linear(ctx_a)) → element-wise scale for ctx_b
        self.gate_a = nn.Linear(bilstm_out, bilstm_out)
        self.gate_b = nn.Linear(bilstm_out, bilstm_out)

        # ── Classification head ───────────────────────────────────────────────
        fused_dim = bilstm_out * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partition (B, T, 147) → pose (B, T, 73) and motion (B, T, 74)."""
        pose = torch.cat([x[..., :63], x[..., 134:144]], dim=-1)    # (B, T, 73)
        motion = torch.cat([x[..., 63:134], x[..., 144:]], dim=-1)  # (B, T, 74)
        return pose, motion

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, T, input_size) where T is the window length
               (32 for IPN 32-frame experiments) and input_size is 147.

        Returns:
            Logits (B, num_classes).
        """
        pose, motion = self._split(x)

        # ── Feature projection ────────────────────────────────────────────────
        pose = self.proj_a(pose)      # (B, T, proj_dim)
        motion = self.proj_b(motion)  # (B, T, proj_dim)

        # ── BiLSTM encoding ───────────────────────────────────────────────────
        out_a, _ = self.lstm_a(pose)    # (B, T, 2H)
        out_b, _ = self.lstm_b(motion)  # (B, T, 2H)

        # ── Pre-attention LayerNorm ───────────────────────────────────────────
        out_a = self.ln_a(out_a)  # (B, T, 2H)
        out_b = self.ln_b(out_b)  # (B, T, 2H)

        # ── Multi-Head Self-Attention (residual + post-LN) ────────────────────
        sa_a, _ = self.mha_a(out_a, out_a, out_a)  # (B, T, 2H)
        out_a = self.ln_post_a(out_a + sa_a)         # residual + LN

        sa_b, _ = self.mha_b(out_b, out_b, out_b)  # (B, T, 2H)
        out_b = self.ln_post_b(out_b + sa_b)         # residual + LN

        # ── Temporal pooling (mean over T) ────────────────────────────────────
        ctx_a = out_a.mean(dim=1)  # (B, 2H)
        ctx_b = out_b.mean(dim=1)  # (B, 2H)

        # ── Cross-stream gating ───────────────────────────────────────────────
        # Compute both gate scales before applying (avoids sequential dependency).
        gate_scale_a = torch.sigmoid(self.gate_a(ctx_b))  # motion → pose gate
        gate_scale_b = torch.sigmoid(self.gate_b(ctx_a))  # pose → motion gate
        ctx_a = ctx_a * gate_scale_a
        ctx_b = ctx_b * gate_scale_b

        # ── Fusion + classification ───────────────────────────────────────────
        fused = torch.cat([ctx_a, ctx_b], dim=-1)  # (B, 4H)
        return self.head(fused)                      # (B, C)
