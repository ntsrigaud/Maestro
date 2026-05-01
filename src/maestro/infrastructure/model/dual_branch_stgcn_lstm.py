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
  Mirrors the paper's SA-ConvLSTM branch, adapted for skeleton input instead
  of raw image frames.  Captures phonological-equivalent features (handshape,
  location, movement, orientation) via:

    (B, T, N, C)
      → Linear per-node projection          (B, T, N, proj_dim)
      → Soft spatial attention over N        (B, T, proj_dim)
      → Temporal Conv1d + BN + ReLU         (B, conv_ch, T)
      → Stacked LSTM                         (B, T, phono_hidden)
      → Bahdanau temporal attention          (B, phono_hidden)

Branch 2 — STGCN (3 blocks, fully-connected graph)
  Three STGCNBlock layers operating on a K=3 partitioned fully-connected
  adjacency (all nodes attend to all nodes; centripetal / centrifugal split
  derived from anatomical BFS distances).

    (B, T, N, C)
      → STGCNBlock(C   → 64,  K=3, residual=True)
      → STGCNBlock(64  → 128, K=3, residual=True)
      → STGCNBlock(128 → 256, K=3, residual=True)
      → mean-pool over N                     (B, T, 256)
      → mean-pool over T                     (B, 256)

Fusion
  cat(branch1, branch2) → Dropout → Linear → (B, num_classes)

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
# Phonological branch components
# ─────────────────────────────────────────────────────────────────────────────


class _SpatialAttnPool(nn.Module):
    """Soft attention over the node dimension → weighted sum.

    At each timestep ``t``, computes a scalar score for every node and
    returns the attention-weighted mean over nodes:

      score(n) = w · x[t, n] + b          (uncalibrated)
      α         = softmax_n(score)
      out[t]    = Σ_n α[n] · x[t, n]

    Args:
        in_dim: Number of feature channels per node.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        w = F.softmax(self.score(x), dim=2)  # (B, T, N, 1)
        return (x * w).sum(dim=2)            # (B, T, C)


class _PhonologicalBranch(nn.Module):
    """Skeleton-adapted Spatial-Attention ConvLSTM branch.

    Captures phonological-equivalent features from skeleton sequences:
      - **Handshape** (finger configuration) through per-node projections.
      - **Location** (wrist/body position) through the spatial attention gate.
      - **Movement** (trajectory) through the temporal convolution.
      - **Temporal context** through the stacked LSTM + attention.

    Args:
        num_nodes:     Number of graph nodes (N).
        num_coords:    Coordinate channels per node (C).
        proj_dim:      Node projection dimension before spatial attention.
        conv_channels: Output channels of the temporal convolution.
        lstm_hidden:   Hidden size of the LSTM.
        lstm_layers:   Number of stacked LSTM layers.
        dropout:       Dropout probability between LSTM layers (≥2 layers).
    """

    def __init__(
        self,
        num_nodes: int,
        num_coords: int,
        proj_dim: int,
        conv_channels: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Step 1: per-node coordinate projection
        self.node_proj = nn.Linear(num_coords, proj_dim)
        self.proj_bn = nn.BatchNorm1d(proj_dim)

        # Step 2: spatial attention over nodes
        self.spatial_attn = _SpatialAttnPool(proj_dim)

        # Step 3: temporal 1-D convolution (mirrors the "Conv" in ConvLSTM)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(proj_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # Step 4: LSTM for long-range temporal modelling
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Step 5: Bahdanau temporal attention over LSTM output
        self.temporal_attn = _BahdanauAttention(lstm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return phonological feature vector ``(B, lstm_hidden)``."""
        B, T, N, _ = x.shape

        # Node projection with BN over channel dim
        h = self.node_proj(x)                          # (B, T, N, proj_dim)
        h = self.proj_bn(h.reshape(-1, h.shape[-1])).reshape(B, T, N, -1)

        # Spatial attention → (B, T, proj_dim)
        h = self.spatial_attn(h)

        # Temporal conv: needs (B, C, T)
        h = self.temporal_conv(h.permute(0, 2, 1))    # (B, conv_ch, T)
        h = h.permute(0, 2, 1)                         # (B, T, conv_ch)

        # LSTM
        lstm_out, _ = self.lstm(h)                     # (B, T, lstm_hidden)

        # Temporal attention → (B, lstm_hidden)
        return self.temporal_attn(lstm_out)


# ─────────────────────────────────────────────────────────────────────────────
# Main dual-branch model
# ─────────────────────────────────────────────────────────────────────────────


class DualBranchSTGCN_LSTM(nn.Module):
    """Dual-branch STGCN-LSTM gesture classifier.

    Implements the two-branch fusion architecture from the reference paper:

    * **Branch 1** (phonological / SA-ConvLSTM): extracts handshape, location,
      movement, and orientation features from the skeleton sequence via a
      spatial-attention ConvLSTM pipeline.
    * **Branch 2** (STGCN, 3 blocks): captures spatio-temporal dependencies
      in the skeleton graph using three stacked STGCNBlocks operating on a
      fully-connected K=3 partitioned adjacency.

    The two branches are concatenated and classified via a linear head.

    Args:
        num_nodes:         Number of graph nodes (default 67).
        num_coords:        Coordinate channels per node (default 3).
        phono_proj_dim:    Node projection dim for Branch 1 (default 128).
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
        phono_proj_dim: int = 128,
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
        self.phono_proj_dim = phono_proj_dim
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
            num_nodes=num_nodes,
            num_coords=num_coords,
            proj_dim=phono_proj_dim,
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
            h = block(h)            # (B, T', N, C_out)
        stgcn_feat = h.mean(dim=(1, 2))  # mean over T and N → (B, C_out)

        # Fusion: concatenate, dropout, classify
        fused = torch.cat([phono_feat, stgcn_feat], dim=-1)  # (B, fusion_dim)
        return self.head(self.dropout(fused))                 # (B, num_classes)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
