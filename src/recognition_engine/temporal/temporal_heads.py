from typing import Dict, Optional

import torch
from torch import nn


class TemporalHeadBase(nn.Module):
    """
    Base interface for temporal heads that consume per-frame embeddings and
    produce (optional) per-frame static predictions and/or sequence-level dynamic predictions.

    Expected input: features of shape (batch, time, dim)
    Expected output: dict with keys:
        - 'static_logits': (batch, time, num_static_classes) or None
        - 'dynamic_logits': (batch, num_dynamic_classes) or None
    """

    def __init__(self, d_model: int, num_static_classes: Optional[int], num_dynamic_classes: Optional[int]):
        super().__init__()
        self.d_model = d_model
        self.num_static_classes = num_static_classes
        self.num_dynamic_classes = num_dynamic_classes

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


class LSTMTemporalHead(TemporalHeadBase):
    """
    Two-layer LSTM temporal head producing per-frame static logits and sequence-level dynamic logits.

    - Input:  x -> (B, T, D)
    - Static: per-frame logits via linear on each hidden state H_t
    - Dynamic: sequence logits via pooled hidden states (mean over time)
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_layers: int,
        num_static_classes: Optional[int] = None,
        num_dynamic_classes: Optional[int] = None,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__(d_model=d_model, num_static_classes=num_static_classes, num_dynamic_classes=num_dynamic_classes)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_size * self.num_directions

        self.static_head = nn.Linear(out_dim, num_static_classes) if num_static_classes else None
        self.dynamic_head = nn.Linear(out_dim, num_dynamic_classes) if num_dynamic_classes else None

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # x: (B, T, D)
        h, _ = self.lstm(x)  # h: (B, T, H*dir)

        static_logits = self.static_head(h) if self.static_head is not None else None

        pooled = h.mean(dim=1)  # (B, H*dir)
        dynamic_logits = self.dynamic_head(pooled) if self.dynamic_head is not None else None

        return {"static_logits": static_logits, "dynamic_logits": dynamic_logits}


class TemporalClassifier(nn.Module):
    """
    Wrapper holding a frozen ViT feature interface and a pluggable temporal head.

    viT_forward: callable that maps images (B, 3, H, W) -> (B, D)
    temporal_head: instance of TemporalHeadBase that maps (B, T, D) -> dict outputs

    forward inputs:
        - images: (B, T, 3, H, W) or precomputed features: (B, T, D) if features_only=True
    """

    def __init__(
        self,
        d_model: int,
        temporal_head: TemporalHeadBase,
        vit_forward: Optional[nn.Module] = None,
        features_only: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_head = temporal_head
        self.vit_forward = vit_forward
        self.features_only = features_only

        if not self.features_only and self.vit_forward is None:
            raise ValueError("vit_forward must be provided when features_only=False")

        # Ensure ViT is frozen by default (caller may override)
        if self.vit_forward is not None:
            for p in self.vit_forward.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _extract_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, T, 3, H, W)
        B, T = images.shape[:2]
        flat = images.view(B * T, *images.shape[2:])
        feats = self.vit_forward(flat)  # (B*T, D)
        feats = feats.view(B, T, -1)
        return feats

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # If x is images: (B, T, 3, H, W) and features_only=False -> extract features
        # If x is features: (B, T, D) and features_only=True -> pass through
        if self.features_only:
            feats = x
        else:
            feats = self._extract_frame_features(x)
        return self.temporal_head(feats)


# Optional Mamba placeholder (to be implemented if/when mamba-ssm is added).
class MambaTemporalHead(TemporalHeadBase):
    def __init__(self, d_model: int, num_static_classes: Optional[int], num_dynamic_classes: Optional[int], **kwargs):
        super().__init__(d_model=d_model, num_static_classes=num_static_classes, num_dynamic_classes=num_dynamic_classes)
        raise NotImplementedError(
            "MambaTemporalHead requires mamba-ssm. Implement this class keeping the same input/output interface."
        )
