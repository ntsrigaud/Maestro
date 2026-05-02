"""Shared STGCN building blocks used by DualBranchSTGCN_LSTM.

This module exposes the following re-usable components:
  - :class:`_GraphConv`         — K-partition spatial graph convolution.
  - :class:`STGCNBlock`         — One SCN→TCN→BN→ReLU→Dropout→+Residual block.
  - :class:`_BahdanauAttention` — Additive temporal attention over LSTM output.

The authoritative model for this project is :class:`DualBranchSTGCN_LSTM`
(see ``dual_branch_stgcn_lstm.py``), which implements the full dual-branch
phonological SA-ConvLSTM architecture from:

  "A Study on the STGCN-LSTM Sign Language Recognition Model Based on
   Phonological Features of Sign Language", IEEE 2025.
   https://ieeexplore.ieee.org/document/10965662

References:
  Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition", AAAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _GraphConv(nn.Module):
    """K-partition graph convolution: ``H' = Σ_k (A_k × H × W_k) + b``.

    Implements the spatial configuration partitioning strategy from Yan et al.
    (AAAI 2018) with K=3 subsets — self-connections, centripetal neighbours,
    and centrifugal neighbours — each gated by an independent weight matrix.

    Args:
        in_channels:  Number of input feature channels per node.
        out_channels: Number of output feature channels per node.
        adjacency:    Adjacency tensor.  Shape ``(K, N, N)`` for K-partition
                      mode, or ``(N, N)`` for single-matrix mode (treated as
                      K=1 for backward compatibility).
    """

    A: torch.Tensor  # (K, N, N) — declared for mypy

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)  # (1, N, N) — backward compat
        K = adjacency.shape[0]
        self.K = K
        self.register_buffer("A", adjacency)
        # One weight matrix per partition.
        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        for k in range(K):
            nn.init.xavier_uniform_(self.weight[k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        B, T, N, C_in = x.shape
        x_flat = x.reshape(B * T, N, C_in)  # (BT, N, Cin)
        # Aggregate neighbours per partition: (K, BT, N, Cin)
        agg = torch.einsum("knm,bmc->kbnc", self.A, x_flat)
        # Apply per-partition weights and sum: (BT, N, Cout)
        out = torch.einsum("kbnc,kco->bno", agg, self.weight) + self.bias
        return out.reshape(B, T, N, -1)  # (B, T, N, Cout)


class STGCNBlock(nn.Module):
    """Spatial Temporal Graph Convolutional block.

    Implements the STGCNBlock as shown in the paper block diagram:

        SCN (spatial graph conv)
        → TCN (single temporal conv)
        → BN → ReLU → Dropout
        → + Residual

    The ReLU and Dropout are applied *before* the residual addition
    (pre-activation style), matching the paper's block diagram.  No ReLU
    is applied after the residual sum.

    Args:
        in_channels:    Node feature channels in.
        out_channels:   Node feature channels out.
        adjacency:      Normalised adjacency ``(N, N)`` tensor.
        temporal_kernel: Temporal convolution kernel size (default 3).
        stride:         Temporal stride (default 1).
        residual:       Whether to add a residual connection (default True).
                        If ``in_channels != out_channels`` a linear projection
                        is used for the skip path.
        dropout:        Dropout probability applied inside the block, after
                        ReLU and before the residual addition (default 0.0).
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = temporal_kernel // 2

        # --- Spatial graph convolution ---
        self.gcn = _GraphConv(in_channels, out_channels, adjacency)
        self.bn_gcn = nn.BatchNorm1d(out_channels)

        # --- Temporal convolution (single-scale, matching paper block diagram) ---
        self.tcn = nn.Conv1d(
            out_channels, out_channels, temporal_kernel, stride=stride, padding=pad
        )
        self.tcn_bn = nn.BatchNorm1d(out_channels)

        # --- Activation and per-block dropout (before residual, per paper diagram) ---
        self.act = nn.ReLU(inplace=True)
        self.block_dropout = nn.Dropout(dropout)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        B, T, N, _ = x.shape

        # 1. Spatial graph conv + BN.
        out = self.gcn(x)  # (B, T, N, C_out)
        out_bn = self.bn_gcn(out.reshape(-1, out.shape[-1])).reshape(B, T, N, -1)

        # 2. Temporal conv — operate per-node: reshape to (B*N, C_out, T).
        t_in = out_bn.permute(0, 2, 3, 1).reshape(B * N, -1, T)  # (B*N, C_out, T)
        t_out = self.tcn(t_in)  # (B*N, C_out, T')
        T_new = t_out.shape[-1]
        t_out = self.tcn_bn(t_out)  # (B*N, C_out, T')
        t_out = t_out.reshape(B, N, -1, T_new).permute(0, 3, 1, 2)  # (B, T', N, C_out)

        # 3. ReLU → Dropout (before residual, per paper block diagram).
        t_out = self.act(t_out)
        t_out = self.block_dropout(t_out)

        # 4. Residual addition (no ReLU after, per paper block diagram).
        if self.residual:
            x_skip = x.permute(0, 2, 3, 1).reshape(B * N, -1, T)  # (B*N, C_in, T)
            x_skip = self.skip_proj(x_skip)  # (B*N, C_out, T')
            x_skip = x_skip.reshape(B, N, -1, T_new).permute(0, 3, 1, 2)
            t_out = t_out + x_skip

        return t_out  # (B, T', N, C_out)


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
