from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from maestro.utils.schema_validation import RuntimeConfig

_logger = logging.getLogger(__name__)


def _build_model_from_artifact(
    model_config: dict[str, Any],
    artifact_type: str,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the correct model class from *artifact_type* / *model_config*.

    Dispatches on ``artifact_type`` (preferred) then falls back to ``model_name``:
    * ``stgcn_lstm_inference``      → :class:`STGCN_LSTM`
    * ``two_stream_lstm_inference`` → :class:`TwoStreamGestureLSTM`
    """
    model_name = str(model_config.get("model_name", ""))
    is_stgcn = artifact_type.startswith("stgcn") or model_name.startswith("stgcn")

    if is_stgcn:
        from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM

        channels_raw = model_config.get("stgcn_channels", [64, 128])
        return STGCN_LSTM(
            num_nodes=int(model_config.get("num_nodes", 67)),
            num_coords=int(model_config.get("num_coords", 3)),
            stgcn_channels=tuple(int(c) for c in channels_raw),
            num_stgcn_layers=int(model_config.get("num_stgcn_layers", 2)),
            lstm_hidden=int(model_config.get("lstm_hidden", 128)),
            lstm_layers=int(model_config.get("lstm_layers", 2)),
            dropout=float(model_config.get("dropout", 0.3)),
            num_classes=int(model_config["num_classes"]),
        ).to(device)

    if artifact_type.startswith("two_stream") or model_name.startswith("two_stream"):
        from maestro.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM

        return TwoStreamGestureLSTM(
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            num_classes=int(model_config["num_classes"]),
        ).to(device)

    raise ValueError(
        f"Unsupported artifact type {artifact_type!r} (model_name={model_name!r}). "
        "Supported types: 'stgcn_lstm_inference', 'two_stream_lstm_inference'."
    )


@dataclass(slots=True)
class InferenceArtifact:
    model: torch.nn.Module
    model_version: str
    class_labels: list[str]
    artifact_path: Path
    # Flat-feature schema fields (two_stream_* models).
    feature_schema_version: str = "unknown"
    feature_dim: int = 0
    window_length: int = 24
    window_step: int = 3
    orientation_normalization: bool = False
    # Graph schema fields (stgcn_* models).
    graph_schema_version: str = "unknown"
    num_nodes: int = 0
    num_coords: int = 0
    adjacency: torch.Tensor | None = None


def load_inference_artifact(
    *,
    artifact_path: str | Path,
    device: torch.device,
) -> InferenceArtifact:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference artifact not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected artifact payload format: {path}")

    model_config = payload.get("model_config")
    class_labels = payload.get("class_labels")
    state_dict = payload.get("model_state_dict")

    if not isinstance(model_config, dict):
        raise ValueError(f"Artifact missing model_config: {path}")
    if not isinstance(class_labels, list) or not class_labels:
        raise ValueError(f"Artifact missing class_labels: {path}")
    if state_dict is None:
        raise ValueError(f"Artifact missing model_state_dict: {path}")

    artifact_type = str(payload.get("artifact_type", ""))
    model = _build_model_from_artifact(model_config, artifact_type, device)
    model.load_state_dict(state_dict)
    model.eval()

    base = InferenceArtifact(
        model=model,
        model_version=str(payload.get("artifact_version", "unknown")),
        class_labels=[str(x) for x in class_labels],
        artifact_path=path,
    )

    is_stgcn = artifact_type.startswith("stgcn") or str(
        model_config.get("model_name", "")
    ).startswith("stgcn")

    if is_stgcn:
        gs = payload.get("graph_schema", {})
        base.graph_schema_version = str(gs.get("graph_schema_version", "unknown"))
        base.num_nodes = int(gs.get("num_nodes", model_config.get("num_nodes", 67)))
        base.num_coords = int(gs.get("num_coords", model_config.get("num_coords", 3)))
        base.window_length = int(gs.get("window_length", 24))
        base.window_step = int(gs.get("window_step", 3))
        raw_adj = gs.get("adjacency")
        if raw_adj is not None:
            base.adjacency = (
                raw_adj if isinstance(raw_adj, torch.Tensor) else torch.as_tensor(raw_adj)
            ).float()
    else:
        fs = payload.get("feature_schema", {})
        if not isinstance(fs, dict):
            raise ValueError(f"Artifact missing feature_schema: {path}")
        base.feature_schema_version = str(fs.get("feature_schema_version", "unknown"))
        base.feature_dim = int(fs.get("feature_dim") or model_config.get("input_size", 0))
        base.window_length = int(fs.get("window_length", 24))
        base.window_step = int(fs.get("window_step", 3))
        base.orientation_normalization = bool(fs.get("orientation_normalization", False))

    return base


def validate_runtime_compatibility(
    *,
    runtime_config: RuntimeConfig,
    artifact: InferenceArtifact,
) -> None:
    if (
        runtime_config.feature_pipeline.orientation_normalization
        != artifact.orientation_normalization
    ):
        raise ValueError(
            "Feature schema mismatch: orientation_normalization differs between runtime "
            f"({runtime_config.feature_pipeline.orientation_normalization}) and model artifact "
            f"({artifact.orientation_normalization})."
        )

    if runtime_config.inference.window_length != artifact.window_length:
        raise ValueError(
            "Inference window_length mismatch: runtime "
            f"{runtime_config.inference.window_length} vs artifact {artifact.window_length}"
        )

    if runtime_config.inference.window_step != artifact.window_step:
        raise ValueError(
            "Inference window_step mismatch: runtime "
            f"{runtime_config.inference.window_step} vs artifact {artifact.window_step}"
        )
