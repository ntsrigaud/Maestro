from __future__ import annotations

import torch
from torch import nn


class _Chomp1d(nn.Module):
    """Remove the right-side padding added by causal convolutions."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class _ResidualTCNBlock(nn.Module):
    """Single TCN residual block with two dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # 1×1 convolution to match channel dimensions for the residual skip.
        self.skip: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.skip(x))


class GestureTCNClassifier(nn.Module):
    """Dilated causal TCN classifier for sequence gesture recognition.

    Parameters
    ----------
    input_size:
        Feature dimensionality of each time step (e.g. 134 or 144).
    num_channels:
        List of output-channel widths for successive TCN blocks.
        Length determines network depth; the receptive field grows with dilation.
    kernel_size:
        Kernel size for every dilated convolution (odd values recommended).
    dropout:
        Dropout rate applied after each activation in the residual blocks.
    num_classes:
        Number of gesture classes.
    """

    def __init__(
        self,
        *,
        input_size: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 8,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        in_ch = input_size
        for level, out_ch in enumerate(num_channels):
            dilation = 2**level
            blocks.append(
                _ResidualTCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → conv expects (B, F, T)
        out = self.network(x.transpose(1, 2))  # (B, C_last, T)
        # Global average pooling over the time dimension.
        context = out.mean(dim=2)  # (B, C_last)
        return self.head(self.dropout(context))
