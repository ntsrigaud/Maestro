"""STGCN-LSTM gesture classifier for graph-schema-v1 input.

Architecture:
  Input  : (B, T, N, C) — T=24 frames, N=67 nodes, C=3 coords
  Block 1: STGCNBlock(in_channels=3,  out_channels=64, temporal_kernel=3)
  Block 2: STGCNBlock(in_channels=64, out_channels=128, temporal_kernel=3)
  Pool   : mean-pool over node dimension → (B, T, 128)
  LSTM   : 2-layer LSTM hidden_size=128 → (B, T, 128)
  Attn   : additive attention → context (B, 128)
  Head   : Dropout + Linear(128, num_classes)

Parameter budget: ≤500 K total.

Graph convolution uses a pre-registered normalised adjacency matrix
(loaded from the checkpoint or passed at construction time) so that
inference is fully self-contained.

References:
  Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition", AAAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _GraphConv(nn.Module):
    """1-hop graph convolution: H' = A_hat × H × W.

    Args:
        in_channels:  Number of input feature channels per node.
        out_channels: Number of output feature channels per node.
        adjacency:    Normalised adjacency matrix ``(N, N)`` — registered as
                      a buffer so it is saved with the model state dict.
    """

    A: torch.Tensor  # normalised adjacency buffer — declared for mypy

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("A", adjacency)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        B, T, N, _ = x.shape
        # Flatten B×T for graph convolution.
        x_flat = x.reshape(B * T, N, -1)  # (B*T, N, C_in)
        # Graph conv: A_hat @ x_flat @ W
        out = torch.bmm(self.A.unsqueeze(0).expand(B * T, -1, -1), x_flat)  # (B*T, N, C_in)
        out = out @ self.weight + self.bias  # (B*T, N, C_out)
        return out.reshape(B, T, N, -1)  # (B, T, N, C_out)


class STGCNBlock(nn.Module):
    """Spatial Temporal Graph Convolutional block.

    Applies multi-scale temporal convolution followed by 1-hop graph
    convolution, batch norm, and an optional residual connection.

    Temporal conv uses two parallel branches: kernel sizes ``temporal_kernel``
    and ``temporal_kernel+2``, concatenated then projected back to
    ``out_channels`` via a 1×1 conv.  This captures both fine- and
    coarse-grained motion cues without doubling the parameter budget.

    Args:
        in_channels:    Node feature channels in.
        out_channels:   Node feature channels out.
        adjacency:      Normalised adjacency ``(N, N)`` tensor.
        temporal_kernel: Base temporal convolution kernel size (default 3).
        stride:         Temporal stride (default 1).
        residual:       Whether to add a residual connection (default True).
                        If ``in_channels != out_channels`` a linear projection
                        is used for the skip path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        *,
        temporal_kernel: int = 3,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        half = out_channels // 2
        pad_a = temporal_kernel // 2
        pad_b = (temporal_kernel + 2) // 2

        # --- Spatial graph convolution ---
        self.gcn = _GraphConv(in_channels, out_channels, adjacency)
        self.bn_gcn = nn.BatchNorm1d(out_channels)

        # --- Multi-scale temporal convolution (two branches → concat → project) ---
        self.tcn_a = nn.Conv1d(out_channels, half, temporal_kernel, stride=stride, padding=pad_a)
        self.tcn_b = nn.Conv1d(
            out_channels, half, temporal_kernel + 2, stride=stride, padding=pad_b
        )
        self.tcn_bn = nn.BatchNorm1d(out_channels)

        # --- Residual ---
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.skip_proj: nn.Module = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.skip_proj = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        B, T, N, _ = x.shape

        # 1. Spatial graph conv.
        out = self.gcn(x)  # (B, T, N, C_out)
        # BatchNorm over (B*T*N) flattened to (B*T*N, C_out).
        out_bn = self.bn_gcn(out.reshape(-1, out.shape[-1])).reshape(B, T, N, -1)

        # 2. Temporal conv — operate per-node: reshape to (B*N, C_out, T).
        t_in = out_bn.permute(0, 2, 3, 1).reshape(B * N, -1, T)  # (B*N, C_out, T)
        ta = self.tcn_a(t_in)  # (B*N, half, T')
        tb = self.tcn_b(t_in)  # (B*N, half, T')
        t_out = torch.cat([ta, tb], dim=1)  # (B*N, C_out, T')
        T_new = t_out.shape[-1]
        t_out = self.tcn_bn(t_out)  # (B*N, C_out, T')
        t_out = t_out.reshape(B, N, -1, T_new).permute(0, 3, 1, 2)  # (B, T', N, C_out)

        # 3. Residual.
        if self.residual:
            # skip path: same temporal conv → (B*N, C_out, T').
            x_skip = x.permute(0, 2, 3, 1).reshape(B * N, -1, T)  # (B*N, C_in, T)
            x_skip = self.skip_proj(x_skip)  # (B*N, C_out, T')
            x_skip = x_skip.reshape(B, N, -1, T_new).permute(0, 3, 1, 2)
            t_out = t_out + x_skip

        return self.act(t_out)  # (B, T', N, C_out)


class _BahdanauAttention(nn.Module):
    """Additive (Bahdanau) temporal attention over an LSTM output sequence."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        return (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)


class STGCN_LSTM(nn.Module):
    """Spatial-Temporal Graph CNN + LSTM gesture classifier.

    The forward pass is::

        (B, T, N, C)
          → STGCNBlock × 2     [spatial + temporal graph features]
          → mean-pool over N   [(B, T, 128)]
          → 2-layer LSTM        [(B, T, lstm_hidden)]
          → attention pool      [(B, lstm_hidden)]
          → Dropout + Linear    [(B, num_classes)]

    The normalised adjacency matrix ``A_hat`` is registered as a buffer and
    embedded in the checkpoint, so no external graph file is needed at
    inference time.

    Args:
        num_nodes:      Number of graph nodes (default 67).
        num_coords:     Coordinate channels per node (default 3).
        stgcn_channels: Tuple of (c1, c2) output channels for the two STGCN
                        blocks (default (64, 128)).
        num_stgcn_layers: Number of STGCN blocks (1 or 2; default 2).
        lstm_hidden:    LSTM hidden state size (default 128).
        lstm_layers:    Number of stacked LSTM layers (default 2).
        dropout:        Dropout probability before the classification head
                        (default 0.3).
        num_classes:    Number of gesture classes (default 8).
        adjacency:      Pre-computed normalised adjacency ``(N, N)`` tensor.
                        If ``None``, constructed from
                        ``maestro.pipeline.graph_schema``.
    """

    def __init__(
        self,
        *,
        num_nodes: int = 67,
        num_coords: int = 3,
        stgcn_channels: tuple[int, ...] = (64, 128),
        num_stgcn_layers: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 8,
        adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.num_coords = num_coords
        self.stgcn_channels = stgcn_channels
        self.num_stgcn_layers = num_stgcn_layers
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        if adjacency is None:
            from maestro.pipeline.graph_schema import build_adjacency, normalise_adjacency

            A = torch.from_numpy(normalise_adjacency(build_adjacency()).astype("float32"))
        else:
            A = adjacency.float()

        # Register so the matrix is saved in the state dict.
        self.register_buffer("adjacency", A)

        # Build STGCN blocks.
        blocks: list[nn.Module] = []
        in_ch = num_coords
        for i in range(num_stgcn_layers):
            out_ch = stgcn_channels[min(i, len(stgcn_channels) - 1)]
            blocks.append(
                STGCNBlock(
                    in_ch,
                    out_ch,
                    A,
                    temporal_kernel=3,
                    residual=(in_ch == out_ch),
                )
            )
            in_ch = out_ch
        self.stgcn_blocks = nn.ModuleList(blocks)

        stgcn_out_dim = stgcn_channels[min(num_stgcn_layers - 1, len(stgcn_channels) - 1)]

        # LSTM.
        self.lstm = nn.LSTM(
            input_size=stgcn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention.
        self.attention = _BahdanauAttention(lstm_hidden)

        # Head.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of gesture windows.

        Args:
            x: Float tensor ``(B, T, N, C)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        # x: (B, T, N, C)
        for block in self.stgcn_blocks:
            x = block(x)  # (B, T', N, C_out)

        # Pool over nodes.
        x = x.mean(dim=2)  # (B, T', C_out)

        # LSTM.
        lstm_out, _ = self.lstm(x)  # (B, T', lstm_hidden)

        # Attention pooling.
        context = self.attention(lstm_out)  # (B, lstm_hidden)

        # Head.
        return self.head(self.dropout(context))  # (B, num_classes)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
