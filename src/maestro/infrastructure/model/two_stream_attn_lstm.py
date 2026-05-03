from __future__ import annotations

import math

import torch
from torch import nn


class EnhancedTwoStreamLSTM(nn.Module):
    """Improved two-stream gesture classifier: BiLSTM + MHA + cross-stream attention.

    Key architectural components:

    1. **Feature projection** — both streams (73 dims and 74 dims) are projected
       to a shared ``proj_dim`` before LSTM processing, making the two BiLSTMs
       fully symmetric and decoupling raw feature width from model capacity.

    2. **Bidirectional LSTMs** — forward *and* backward temporal context over
       the fixed-length window, with inter-layer dropout for regularisation.

    3. **Sinusoidal positional encoding** — added to the BiLSTM output before
       self-attention, giving the attention heads explicit knowledge of each
       frame's temporal position within the window.  This is critical for
       gesture recognition where *when* a frame occurs carries semantic meaning.

    4. **Output dropout** — applied to the position-encoded BiLSTM output before
       self-attention to regularise the representations entering the attention
       mechanism.

    5. **Multi-Head Self-Attention (Q/K/V)** — each of the T output frames
       attends to all other frames across ``num_heads`` independent subspaces,
       yielding a richer temporal summary before mean-pooling.  Pre-norm
       residual structure (LayerNorm → MHA → residual + LN) for stability.

    6. **Cross-stream cross-attention** — replaces the previous scalar sigmoid
       gate.  After temporal mean-pooling, each stream's context vector queries
       the *other stream's temporal sequence* via a full MHA cross-attention
       block.  This lets pose features focus on the specific motion frames most
       relevant to the gesture and vice-versa, rather than receiving only a
       blended scalar from a single linear gate.

    7. **Deeper MLP head** — ``LayerNorm → Linear → GELU → Dropout → Linear``.

    Feature partition (feature-schema-v5, 147 dims total):
        Stream A — Pose / Shape (73 dims)
            dims   0:63   positions       (21 wrist-relative landmarks × 3)
            dims 134:144  joint_angles    (10 MCP + PIP joint angles)

        Stream B — Motion / Dynamics (74 dims)
            dims  63:68   fingertip_spread (5 inter-fingertip distances)
            dims  68:71   wrist_trajectory (3 net wrist displacement)
            dims  71:134  velocity         (21 per-landmark velocity × 3)
            dims 144:147  wrist_vel_raw    (3 camera-normalised wrist velocity)

    Parameter budget (proj_dim=128, hidden_size=128, num_layers=2, num_heads=8):
        proj_a / proj_b         2 × ~10 K             ≈   20 K
        BiLSTM-A                2-layer bidir (128)    ≈  656 K
        BiLSTM-B                (identical)            ≈  656 K
        MHA-A / MHA-B (self)    2 × (8 × 256² + …)    ≈  526 K
        Cross-attn A→B / B→A   2 × (same dims)        ≈  526 K
        LN × 8                  negligible             ≈    1 K
        Head MLP                512 → 256 → C          ≈  134 K
        Total                                          ≈ 2.52 M

    Parameter budget (proj_dim=64, hidden_size=64, num_layers=2, num_heads=4):
        (smaller reference configuration)
        proj + BiLSTMs + self-attn (original)          ≈  541 K
        Cross-attn A→B / B→A   2 × (4 × 128² + …)    ≈  132 K
        Total                                          ≈  673 K
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
        max_seq_len: int = 128,
    ) -> None:
        """Initialise the enhanced two-stream LSTM.

        Args:
            input_size: Total feature dimensionality (147 for feature-schema-v5);
                stored for artifact serialisation only.
            hidden_size: Hidden units *per direction per stream*.  BiLSTM output
                width = ``2 × hidden_size``.  Recommended: 128.
            num_layers: LSTM depth.  Dropout is applied between layers when
                ``num_layers > 1``.  Recommended: 2.
            dropout: Dropout rate applied between LSTM layers, on BiLSTM output
                (before attention), inside MHA, and in the head Dropout.
            num_classes: Output vocabulary size.
            num_heads: Number of MHA heads.  Must divide ``2 × hidden_size``
                evenly.  Default: 4 (use 8 for hidden_size=128).
            proj_dim: Output dimension of the per-stream feature projection.
                Should equal or exceed ``hidden_size``.  Recommended: 128.
            max_seq_len: Upper bound on sequence length for pre-building the
                sinusoidal positional encoding buffer.  Default: 128 (covers
                all IPN window configurations).
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

        # ── Sinusoidal positional encoding (no learnable params) ──────────────
        self.register_buffer(
            "_pe",
            self._sinusoidal_pe(max_seq_len, bilstm_out),  # (1, max_seq_len, bilstm_out)
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

        # ── Output dropout (applied after PE, before attention) ───────────────
        self.out_drop = nn.Dropout(p=dropout)

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

        # ── Cross-stream cross-attention ──────────────────────────────────────
        # cross_attn_a: ctx_a (1 query) attends to out_b (T keys/values)
        # cross_attn_b: ctx_b (1 query) attends to out_a (T keys/values)
        # This replaces the scalar sigmoid gate with a full attention mechanism
        # so each stream can focus on the specific *frames* of the other stream
        # most relevant to the current gesture.
        self.cross_attn_a = nn.MultiheadAttention(**_mha_kw)
        self.cross_attn_b = nn.MultiheadAttention(**_mha_kw)

        # Post-cross-attention LayerNorm (residual on pooled context)
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
    def _sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        """Build a ``(1, max_len, dim)`` sinusoidal positional encoding tensor.

        Uses the standard Transformer formulation:
            PE[pos, 2i]   = sin(pos / 10000^(2i / dim))
            PE[pos, 2i+1] = cos(pos / 10000^(2i / dim))
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        half = dim // 2
        div_term = torch.exp(
            torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )  # (dim//2,)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, dim)

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
        pose   = self.proj_a(pose)    # (B, T, proj_dim)
        motion = self.proj_b(motion)  # (B, T, proj_dim)

        # ── BiLSTM encoding ───────────────────────────────────────────────────
        out_a, _ = self.lstm_a(pose)    # (B, T, 2H)
        out_b, _ = self.lstm_b(motion)  # (B, T, 2H)

        # ── Sinusoidal positional encoding + output dropout ───────────────────
        T = out_a.size(1)
        pe = self._pe[:, :T, :]          # (1, T, 2H)  — type: ignore[index]
        out_a = self.out_drop(out_a + pe)
        out_b = self.out_drop(out_b + pe)

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

        # ── Cross-stream cross-attention ──────────────────────────────────────
        # Each stream's mean-pooled context (single query) attends to all T
        # frames of the *other* stream, allowing fine-grained temporal focus.
        # Shape of query: (B, 1, 2H);  key/value: (B, T, 2H) → (B, 1, 2H) out
        ca_a, _ = self.cross_attn_a(ctx_a.unsqueeze(1), out_b, out_b)  # (B, 1, 2H)
        ca_b, _ = self.cross_attn_b(ctx_b.unsqueeze(1), out_a, out_a)  # (B, 1, 2H)

        # Residual + LN on pooled context (Transformer-style sublayer)
        ctx_a = self.ln_cross_a(ctx_a + ca_a.squeeze(1))  # (B, 2H)
        ctx_b = self.ln_cross_b(ctx_b + ca_b.squeeze(1))  # (B, 2H)

        # ── Fusion + classification ───────────────────────────────────────────
        fused = torch.cat([ctx_a, ctx_b], dim=-1)  # (B, 4H)
        return self.head(fused)                      # (B, C)
