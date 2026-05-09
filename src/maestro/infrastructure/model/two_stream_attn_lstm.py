from __future__ import annotations

import torch
from torch import nn


class EnhancedTwoStreamLSTM(nn.Module):
    """Two-stream gesture classifier: BiLSTM + MHA self-attention + MLP cross-stream gate.

    Key architectural components:

    1. **Feature projection** — both streams (73 dims and 74 dims) are projected
       to a shared ``proj_dim`` before LSTM processing, making the two BiLSTMs
       fully symmetric and decoupling raw feature width from model capacity.

    2. **Bidirectional LSTMs** — forward *and* backward temporal context over
       the fixed-length window, with inter-layer dropout for regularisation.
       BiLSTMs are inherently sequential processors; no positional encoding is
       needed (and would conflict with the LSTM's own temporal encoding).

    3. **Multi-Head Self-Attention (Q/K/V)** — each of the T output frames
       attends to all other frames across ``num_heads`` independent subspaces,
       yielding a richer temporal summary before pooling.  Pre-norm
       residual structure (LayerNorm → MHA → residual + post-LN) for stability.

    4. **Mean + Max temporal pooling** — element-wise sum of temporal mean and
       temporal max.  Mean captures sustained gesture shape; max captures peak
       activity from transient events (click, double-click).  A LayerNorm is
       applied after pooling to handle the variance introduced by combining
       two statistics.

    5. **2-layer MLP cross-stream gate** — replaces the previous full MHA
       cross-attention block (which used a degenerate mean-pooled query).
       Each stream's pooled context is gated by a small MLP reading from the
       *other* stream's context, providing compact cross-modal recalibration
       (~37 K params per gate vs ~263 K for full MHA cross-attention).

    6. **Deeper MLP head** — ``LayerNorm → Linear → GELU → Dropout → Linear``.

    Feature partition (feature-schema-v5, 147 dims total):
        Stream A — Pose / Shape (73 dims)
            dims   0:63   positions       (21 wrist-relative landmarks × 3)
            dims 134:144  joint_angles    (10 MCP + PIP joint angles)

        Stream B — Motion / Dynamics (74 dims)
            dims  63:68   fingertip_spread (5 inter-fingertip distances)
            dims  68:71   wrist_trajectory (3 net wrist displacement)
            dims  71:134  velocity         (21 per-landmark velocity × 3)
            dims 144:147  wrist_vel_raw    (3 camera-normalised wrist velocity)

    Parameter budget (proj_dim=96, hidden_size=96, num_layers=2, num_heads=8, 10 classes):
        proj_a / proj_b           2 × ~7 K              ≈   14 K
        BiLSTM-A                  2-layer bidir (96)     ≈  370 K
        BiLSTM-B                  (identical)            ≈  370 K
        MHA-A / MHA-B (self)      2 × (8 × 192² + …)    ≈  296 K
        pool_ln_a / pool_ln_b     2 × LN(192)            ≈    1 K
        gate_a / gate_b           2 × MLP(192→96→192)    ≈   74 K
        ln_cross_a / ln_cross_b   2 × LN(192)            ≈    1 K
        Head MLP                  384 → 192 → C          ≈   76 K
        Total                                            ≈ 1.20 M

    Parameter budget (proj_dim=64, hidden_size=64, num_layers=2, num_heads=4, 10 classes):
        (small reference configuration)
        proj + BiLSTMs + self-attn + gates + head       ≈  542 K
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
            input_size: Total feature dimensionality (147 for feature-schema-v5);
                stored for artifact serialisation only.
            hidden_size: Hidden units *per direction per stream*.  BiLSTM output
                width = ``2 × hidden_size``.  Recommended: 96.
            num_layers: LSTM depth.  Dropout is applied between layers when
                ``num_layers > 1``.  Recommended: 2.
            dropout: Dropout rate applied between LSTM layers and in the head
                Dropout.  MHA uses this as ``dropout`` on attention weights.
            num_classes: Output vocabulary size.
            num_heads: Number of MHA heads.  Must divide ``2 × hidden_size``
                evenly.  Default: 4 (use 8 for hidden_size ≥ 96).
            proj_dim: Output dimension of the per-stream feature projection.
                Should equal ``hidden_size``.  Recommended: 96.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
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

        # ── Post-pool LayerNorm ───────────────────────────────────────────────
        # Mean + Max pooling combines two statistics; LN re-centres the result.
        self.pool_ln_a = nn.LayerNorm(bilstm_out)
        self.pool_ln_b = nn.LayerNorm(bilstm_out)

        # ── Cross-stream MLP gate ─────────────────────────────────────────────
        # gate_a reads ctx_b and produces element-wise sigmoid gates for ctx_a.
        # A 2-layer MLP with a bottleneck (bilstm_out → bilstm_out//2 → bilstm_out)
        # provides non-linear cross-modal recalibration at ~37 K params each
        # (vs ~263 K for full MHA cross-attention).
        _gate_hidden = bilstm_out // 2
        self.gate_a = nn.Sequential(
            nn.Linear(bilstm_out, _gate_hidden),
            nn.Tanh(),
            nn.Linear(_gate_hidden, bilstm_out),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(bilstm_out, _gate_hidden),
            nn.Tanh(),
            nn.Linear(_gate_hidden, bilstm_out),
            nn.Sigmoid(),
        )

        # Post-gate LayerNorm (applied to gated residual output)
        self.ln_cross_a = nn.LayerNorm(bilstm_out)
        self.ln_cross_b = nn.LayerNorm(bilstm_out)

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
        pose = torch.cat([x[..., :63], x[..., 134:144]], dim=-1)  # (B, T, 73)
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
        pose = self.proj_a(pose)  # (B, T, proj_dim)
        motion = self.proj_b(motion)  # (B, T, proj_dim)

        # ── BiLSTM encoding ───────────────────────────────────────────────────
        out_a, _ = self.lstm_a(pose)  # (B, T, 2H)
        out_b, _ = self.lstm_b(motion)  # (B, T, 2H)

        # ── Pre-attention LayerNorm ───────────────────────────────────────────
        out_a = self.ln_a(out_a)  # (B, T, 2H)
        out_b = self.ln_b(out_b)  # (B, T, 2H)

        # ── Multi-Head Self-Attention (residual + post-LN) ────────────────────
        sa_a, _ = self.mha_a(out_a, out_a, out_a)  # (B, T, 2H)
        out_a = self.ln_post_a(out_a + sa_a)  # residual + LN

        sa_b, _ = self.mha_b(out_b, out_b, out_b)  # (B, T, 2H)
        out_b = self.ln_post_b(out_b + sa_b)  # residual + LN

        # ── Temporal pooling: mean + max ──────────────────────────────────────
        # Element-wise sum of temporal mean and temporal max:
        #   mean → captures sustained gesture shape (e.g. swipe direction)
        #   max  → captures peak activity (e.g. brief click / tap event)
        ctx_a = self.pool_ln_a(out_a.mean(dim=1) + out_a.max(dim=1).values)  # (B, 2H)
        ctx_b = self.pool_ln_b(out_b.mean(dim=1) + out_b.max(dim=1).values)  # (B, 2H)

        # ── Cross-stream MLP gating ───────────────────────────────────────────
        # gate_a reads ctx_b and computes element-wise gates for ctx_a.
        # Gated residual + LN preserves the original representation while
        # selectively emphasising features relevant to the other stream.
        gate_a = self.gate_a(ctx_b)  # (B, 2H) in [0, 1]
        ctx_a = self.ln_cross_a(ctx_a * gate_a + ctx_a)  # gated residual + LN

        gate_b = self.gate_b(ctx_a)  # (B, 2H) in [0, 1]
        ctx_b = self.ln_cross_b(ctx_b * gate_b + ctx_b)  # gated residual + LN

        # ── Fusion + classification ───────────────────────────────────────────
        fused = torch.cat([ctx_a, ctx_b], dim=-1)  # (B, 4H)
        return self.head(fused)  # (B, C)
