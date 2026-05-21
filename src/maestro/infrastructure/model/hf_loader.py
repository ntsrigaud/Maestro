"""Hugging Face Hub inference artifact loader.

Downloads (or returns cached) a gesture control inference artifact from a
Hugging Face Hub model repository and wraps it as an :class:`InferenceArtifact`.

After the first download the artifact is served from the local HF cache
(``~/.cache/huggingface/hub`` by default), so subsequent calls require no
network round-trip as long as the same ``revision`` is requested.

Example::

    from maestro.infrastructure.model.hf_loader import load_artifact_from_hub
    import torch

    artifact = load_artifact_from_hub(
        repo_id="ntsrigaud/gesture-control-lstm",
        filename="lstm_v1_inference.pt",
        device=torch.device("cpu"),
    )
"""

from __future__ import annotations

from pathlib import Path

import torch

from maestro.infrastructure.model.checkpoint_loader import (
    InferenceArtifact,
    load_inference_artifact,
)

try:
    from huggingface_hub import hf_hub_download

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]
    _HF_AVAILABLE = False


def load_artifact_from_hub(
    *,
    repo_id: str,
    filename: str,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    device: torch.device,
) -> InferenceArtifact:
    """Download (or return cached) an inference artifact from Hugging Face Hub.

    Parameters
    ----------
    repo_id:
        Hugging Face Hub repository identifier, e.g. ``"ntsrigaud/gesture-control-lstm"``.
    filename:
        Name of the ``.pt`` artifact file inside the repository.
    revision:
        Git branch, tag, or commit hash to resolve.  Defaults to ``"main"``.
        Pin to a tag (e.g. ``"v1.0.0"``) for reproducible deployments.
    cache_dir:
        Override the HF Hub cache directory.  ``None`` uses the default
        ``~/.cache/huggingface/hub``.
    device:
        Kept for API compatibility. Runtime inference is CPU-only and model
        weights are always loaded on CPU.

    Returns
    -------
    InferenceArtifact
        Fully initialised artifact with model in eval mode.
    """
    if not _HF_AVAILABLE or hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required for HF model loading. "
            "Install it with: pip install huggingface_hub"
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return load_inference_artifact(artifact_path=Path(local_path), device=torch.device("cpu"))
