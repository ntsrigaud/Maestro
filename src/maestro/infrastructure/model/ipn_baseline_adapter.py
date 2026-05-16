from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from maestro.infrastructure.model.IPN_models import c3d as ipn_c3d
from maestro.infrastructure.model.IPN_models import c3d_BN as ipn_c3d_bn
from maestro.infrastructure.model.IPN_models import mobilenetv2 as ipn_mobilenetv2
from maestro.infrastructure.model.IPN_models import resnet as ipn_resnet
from maestro.infrastructure.model.IPN_models import resnetl as ipn_resnetl
from maestro.infrastructure.model.IPN_models import resnext as ipn_resnext
from maestro.infrastructure.model.IPN_models import shufflenetv2 as ipn_shufflenetv2


@dataclass(slots=True)
class IPNBaselineAdapterConfig:
    input_size: int
    num_classes: int
    sequence_length: int
    sample_size: int = 32
    width_mult: float = 1.0
    shortcut_type: str = "B"
    cardinality: int = 32


def _build_ipn_backbone(backbone_name: str, cfg: IPNBaselineAdapterConfig) -> nn.Module:
    if backbone_name == "c3d":
        return ipn_c3d.c3d_v1(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "c3d_bn":
        return ipn_c3d_bn.c3d_v1(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "mobilenetv2":
        return ipn_mobilenetv2.mob_v2(
            num_classes=cfg.num_classes,
            sample_size=cfg.sample_size,
            width_mult=cfg.width_mult,
        )
    if backbone_name == "resnet10":
        return ipn_resnet.resnet10(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            shortcut_type=cfg.shortcut_type,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "resnet18":
        return ipn_resnet.resnet18(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            shortcut_type=cfg.shortcut_type,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "resnetl10":
        return ipn_resnetl.resnetl10(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            shortcut_type=cfg.shortcut_type,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "resnext101":
        return ipn_resnext.resnet101(
            sample_size=cfg.sample_size,
            sample_duration=cfg.sequence_length,
            shortcut_type=cfg.shortcut_type,
            cardinality=cfg.cardinality,
            num_classes=cfg.num_classes,
        )
    if backbone_name == "shufflenetv2":
        return ipn_shufflenetv2.shf_v2(
            num_classes=cfg.num_classes,
            sample_size=cfg.sample_size,
            width_mult=cfg.width_mult,
        )

    supported = [
        "c3d",
        "c3d_bn",
        "mobilenetv2",
        "resnet10",
        "resnet18",
        "resnetl10",
        "resnext101",
        "shufflenetv2",
    ]
    raise ValueError(f"Unsupported IPN baseline backbone {backbone_name!r}. Supported: {supported}")


class IPNFeatureToVideoAdapter(nn.Module):
    """Adapter that maps feature sequences (B, T, F) to 3D-CNN input (B, C, T, H, W)."""

    def __init__(
        self,
        *,
        backbone_name: str,
        config: IPNBaselineAdapterConfig,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.input_size = config.input_size
        self.num_classes = config.num_classes
        self.sequence_length = config.sequence_length
        self.sample_size = config.sample_size

        projection_dim = 3 * self.sample_size * self.sample_size
        self.project = nn.Sequential(
            nn.Linear(self.input_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.backbone = _build_ipn_backbone(backbone_name, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {tuple(x.shape)}")

        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.input_size:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_size}, got {feature_dim}"
            )

        projected = self.project(x)
        frames = projected.view(batch, seq_len, 3, self.sample_size, self.sample_size)
        video = frames.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(video)
