from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalHeadBase(nn.Module):
    """
    Base interface for temporal heads that consume per-frame embeddings and
    produce (optional) per-frame static predictions and/or sequence-level dynamic predictions.

    Input:
        x: (B, T, D)
        lengths (optional): (B,)

    Output:
        {
            'static_logits': (B, T, C_static) or None,
            'dynamic_logits': (B, C_dynamic) or None
        }
    """

    def __init__(
        self,
        d_model: int,
        num_static_classes: Optional[int],
        num_dynamic_classes: Optional[int],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_static_classes = num_static_classes
        self.num_dynamic_classes = num_dynamic_classes

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


class LSTMTemporalHead(TemporalHeadBase):
    """
    LSTM temporal head for gesture modeling.

    - Input:  (B, T, D)
    - Static: per-frame logits from LSTM hidden states
    - Dynamic: sequence-level logits from final hidden state
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_layers: int = 1,
        num_static_classes: Optional[int] = None,
        num_dynamic_classes: Optional[int] = None,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__(
            d_model=d_model,
            num_static_classes=num_static_classes,
            num_dynamic_classes=num_dynamic_classes,
        )

        self.hidden_size = hidden_size
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

        self.static_head = (
            nn.Linear(out_dim, num_static_classes)
            if num_static_classes is not None
            else None
        )

        self.dynamic_head = (
            nn.Linear(out_dim, num_dynamic_classes)
            if num_dynamic_classes is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        x: (B, T, D)
        lengths: (B,) valid frame counts per sequence
        """

        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            h, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, (h_n, _) = self.lstm(x)

        # h: (B, T, H * dir)
        static_logits = self.static_head(h) if self.static_head is not None else None

        # Extract final temporal representation
        if self.dynamic_head is not None:
            if self.bidirectional:
                # Last layer: forward and backward
                forward_final = h_n[-2]  # (B, H)
                backward_final = h_n[-1]  # (B, H)
                pooled = torch.cat([forward_final, backward_final], dim=-1)
            else:
                pooled = h_n[-1]  # (B, H)

            dynamic_logits = self.dynamic_head(pooled)
        else:
            dynamic_logits = None

        return {
            "static_logits": static_logits,
            "dynamic_logits": dynamic_logits,
        }


class TemporalClassifier(nn.Module):
    """
    Wrapper combining a ViT frame encoder and a temporal head.

    Input:
        - images: (B, T, 3, H, W)
        - OR features: (B, T, D)

    Output:
        dict from temporal head
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

        if self.vit_forward is not None:
            for p in self.vit_forward.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _extract_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        B, T = images.shape[:2]
        flat = images.view(B * T, *images.shape[2:])
        feats = self.vit_forward(flat)  # (B*T, D)
        return feats.view(B, T, -1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.features_only:
            feats = x
        else:
            feats = self._extract_frame_features(x)

        return self.temporal_head(feats, lengths)
