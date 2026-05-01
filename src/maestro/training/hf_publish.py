"""Hugging Face Hub publishing for trained gesture recognition models.

Usage
-----
    from maestro.training.hf_publish import publish_to_huggingface

    url = publish_to_huggingface(
        result=train_result,
        config=pipeline_config,
        test_report=test_evaluation_report,
        inference_artifact_path="models/exported/lstm_v1_20260417T052358Z_inference.pt",
        repo_id="your-username/maestro-lstm",
        token="hf_...",        # or set HF_TOKEN env var
        private=True,
    )
    print(f"Published: {url}")
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, login

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False

from maestro.training.evaluate import EvaluationReport
from maestro.training.train_lstm import TrainingPipelineConfig, TrainingResult

# ---------------------------------------------------------------------------
# Feature schema description embedded in the model card
# ---------------------------------------------------------------------------
_FEATURE_SCHEMA_DESCRIPTION = """\
| Block | Dims | Description |
|-------|------|-------------|
| `position` | 0–62 | 21 wrist-relative, scale-normalised landmark positions (x, y, z) |
| `fingertip_spread` | 63–67 | 5 inter-fingertip Euclidean distances |
| `wrist_trajectory` | 68–70 | Net wrist displacement from oldest frame in the window |
| `velocity` | 71–133 | 21 per-landmark wrist-relative velocity vectors (Δposition per unit time) |
| `joint_angles` | 134–143 | 10 MCP + PIP joint angles in radians |
| `wrist_vel_raw` | 144–146 | Camera-normalised wrist velocity (x, y, z) — key directional signal |
"""

_CLASS_DESCRIPTIONS: dict[str, str] = {
    "open_palm_hold": "Static open palm facing camera",
    "swipe_right": "Horizontal swipe from left to right",
    "swipe_left": "Horizontal swipe from right to left",
    "swipe_up": "Vertical swipe upward",
    "swipe_down": "Vertical swipe downward",
    "zoom_in": "Pinch-open (spread fingers away from each other)",
    "zoom_out": "Pinch-close (bring fingers together)",
    "unknown": "Background / transition / no gesture",
}

_GRAPH_SCHEMA_DESCRIPTION = """\
| Block | Nodes | Coords | Description |
|-------|-------|--------|-------------|
| `right_hand` | 0–20 | x, y, z | MediaPipe right-hand landmarks (21 nodes) |
| `left_hand` | 21–41 | x, y, z | MediaPipe left-hand landmarks (21 nodes) |
| `pose` | 42–52 | x, y, z | 11 selected MediaPipe Pose landmarks (wrists, elbows, shoulders, ears, eyes, nose) |
| `face` | 53–66 | x, y, z | 14 selected MediaPipe Face Mesh contour points |

Coordinates are wrist-anchor normalised: translated by the mean of both wrist positions
and scaled by the inter-wrist distance.
"""

_STGCN_MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: mit
tags:
- gesture-recognition
- hand-gesture
- pytorch
- mediapipe
- temporal-model
- graph-neural-network
- stgcn
datasets:
- IPN-Hand
metrics:
- accuracy
- f1
model-index:
- name: {model_version}
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: {test_accuracy:.4f}
    - type: f1
      value: {test_f1:.4f}
---

# {model_version}

A real-time hand gesture classifier trained on the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe Holistic for landmark extraction.

## Model Description

- **Architecture**: {architecture}
- **Parameters**: {num_params:,}
- **Input**: `(batch, 24, 67, 3)` — 24-frame sliding window, 67-node graph, 3 coordinates
- **Output**: Softmax logits over {num_classes} gesture classes
- **Graph schema**: `{graph_schema_version}` — 67 nodes (hands + pose + face)

## Gesture Classes

| Class | Description |
|-------|-------------|
{class_table}

## Graph Schema (`{graph_schema_version}`)

{graph_schema_description}

## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{model_version}_inference.pt",
)

# Load the artifact (includes model, class labels, and graph schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Extract 67-node holistic keypoints via HolisticLandmarkExtractor,
# fill a 24-frame GraphSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=24, N=67, C=3)
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: IPN Hand database — multi-subject, multi-session gesture recordings
- **Used classes**: {num_classes} ({num_classes_active} active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (group-aware split by recording session)
- **Graph**: 67 nodes — 42 hand + 11 pose + 14 face contour landmarks
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.003;
  label-aware offline horizontal mirror (swipe_left ↔ swipe_right, node-block swap)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | {architecture} |
| STGCN channels | {stgcn_channels} |
| LSTM hidden | {lstm_hidden} |
| LSTM layers | {lstm_layers} |
| Dropout | {dropout} |
| Learning rate | {learning_rate} |
| Batch size | {batch_size} |
| Max epochs | {epochs} |
| Early stopping patience | {patience} |
| Class weighting | ✓ inverse-frequency |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.1%} |
| Macro F1 | {test_f1:.1%} |

### Per-Class Recall

| Class | Recall |
|-------|--------|
{per_class_table}

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_ipn.yaml`.
- Requires **mediapipe>=0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.

## Environmental Impact

Training was performed on CPU. Estimated training time: ~15 minutes.
Estimated CO₂ equivalent: negligible (<0.005 kg CO₂eq).

---

*Generated by the Maestro training pipeline on {created_at}.*
"""

_TWO_STREAM_MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: mit
tags:
- gesture-recognition
- hand-gesture
- pytorch
- mediapipe
- temporal-model
- lstm
datasets:
- IPN-Hand
metrics:
- accuracy
- f1
model-index:
- name: {model_version}
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: {test_accuracy:.4f}
    - type: f1
      value: {test_f1:.4f}
---

# {model_version}

A real-time hand gesture classifier trained on the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe for landmark extraction.

## Model Description

- **Architecture**: {architecture}
- **Parameters**: {num_params:,}
- **Input**: `(batch, 24, {feature_dim})` — 24-frame sliding window at 30 FPS ≈ 800 ms
- **Output**: Softmax logits over {num_classes} gesture classes
- **Inference latency**: ~0.5 ms per call (CPU, single sample)
- **Feature schema**: `{feature_schema_version}`

## Gesture Classes

| Class | Description |
|-------|-------------|
{class_table}

## Feature Schema (`{feature_schema_version}`)

{feature_schema_description}

## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{model_version}_inference.pt",
)

# Load the artifact (includes model, class labels, and feature schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Build a {feature_dim}-dim feature vector using LandmarkFeatureTransformer
# and fill a 24-frame SlidingWindowSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=24, F={feature_dim})
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: IPN Hand database — multi-subject, multi-session gesture recordings
- **Used classes**: {num_classes} ({num_classes_active} active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (group-aware split by recording session)
- **Sequences**: ~22,973 sliding windows × 24 frames × {feature_dim} features
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.005;
  label-aware offline horizontal mirror (swipe_left ↔ swipe_right)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | {architecture} |
| Input size | {feature_dim} |
| Hidden size | {hidden_size} |
| Num layers | {num_layers} |
| Dropout | {dropout} |
| Learning rate | {learning_rate} |
| Batch size | {batch_size} |
| Max epochs | {epochs} |
| Early stopping patience | {patience} |
| Label smoothing | 0.1 |
| Class weighting | ✓ inverse-frequency |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.1%} |
| Macro F1 | {test_f1:.1%} |

### Per-Class Recall

| Class | Recall |
|-------|--------|
{per_class_table}

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_ipn.yaml`.
- Requires **mediapipe==0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.
- Performance was measured on a held-out test split from the same dataset; real-world
  generalisation may differ.

## Environmental Impact

Training was performed on CPU. Estimated training time: ~5 minutes.
Estimated CO₂ equivalent: negligible (<0.001 kg CO₂eq).

---

*Generated by the Maestro training pipeline on {created_at}.*
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_class_table(class_labels: list[str]) -> str:
    rows = []
    for cls in class_labels:
        desc = _CLASS_DESCRIPTIONS.get(cls, "—")
        rows.append(f"| `{cls}` | {desc} |")
    return "\n".join(rows)


def _build_per_class_table(per_class_recall: dict[str, float]) -> str:
    rows = [f"| `{cls}` | {recall:.1%} |" for cls, recall in per_class_recall.items()]
    return "\n".join(rows)


def _build_model_card(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    repo_id: str = "ntsrigaud/maestro-lstm",
) -> str:
    is_stgcn = config.model.model_name.startswith("stgcn")
    num_params = _count_parameters(result.model)
    num_classes_active = len(result.class_labels) - 1  # subtract unknown
    created_at = datetime.now(UTC).strftime("%Y-%m-%d")

    if is_stgcn:
        architecture = (
            f"STGCN-LSTM "
            f"(channels={getattr(config.model, 'stgcn_channels', [64, 128])}, "
            f"lstm_hidden={getattr(config.model, 'lstm_hidden', 128)}, "
            f"layers={getattr(config.model, 'lstm_layers', 2)})"
        )
        return _STGCN_MODEL_CARD_TEMPLATE.format(
            model_version=result.model_version,
            repo_id=repo_id,
            architecture=architecture,
            num_params=num_params,
            num_classes=config.model.num_classes,
            num_classes_active=num_classes_active,
            graph_schema_version=getattr(config.export, "graph_schema_version", "graph-schema-v1"),
            graph_schema_description=_GRAPH_SCHEMA_DESCRIPTION,
            class_table=_build_class_table(result.class_labels),
            test_accuracy=test_report.accuracy,
            test_f1=test_report.macro_f1,
            stgcn_channels=getattr(config.model, "stgcn_channels", [64, 128]),
            lstm_hidden=getattr(config.model, "lstm_hidden", 128),
            lstm_layers=getattr(config.model, "lstm_layers", 2),
            dropout=config.model.dropout,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
            per_class_table=_build_per_class_table(test_report.per_class_recall),
            created_at=created_at,
        )

    # Legacy two-stream LSTM path
    architecture = (
        f"Two-stream LSTM with temporal attention "
        f"(hidden_size={config.model.hidden_size}/stream, layers={config.model.num_layers})"
    )
    return _TWO_STREAM_MODEL_CARD_TEMPLATE.format(
        model_version=result.model_version,
        repo_id=repo_id,
        architecture=architecture,
        num_params=num_params,
        feature_dim=config.model.input_size,
        num_classes=config.model.num_classes,
        num_classes_active=num_classes_active,
        feature_schema_version=config.export.feature_schema_version,
        feature_schema_description=_FEATURE_SCHEMA_DESCRIPTION,
        class_table=_build_class_table(result.class_labels),
        test_accuracy=test_report.accuracy,
        test_f1=test_report.macro_f1,
        hidden_size=getattr(config.model, "hidden_size", "N/A"),
        num_layers=getattr(config.model, "num_layers", "N/A"),
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        per_class_table=_build_per_class_table(test_report.per_class_recall),
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model_card(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    repo_id: str = "ntsrigaud/maestro-lstm",
) -> str:
    """Return the rendered Markdown model card string without uploading."""
    return _build_model_card(result=result, config=config, test_report=test_report, repo_id=repo_id)


def publish_to_huggingface(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    inference_artifact_path: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
    commit_message: str | None = None,
) -> str:
    """Publish a trained model to the Hugging Face Hub.

    Uploads three artefacts in two commits:
      1. ``README.md``     — rendered model card (YAML front-matter + Markdown body)
      2. ``config.json``   — structured metadata (model config, feature schema, metrics)
      3. ``<version>_inference.pt`` — the exported PyTorch inference artefact

    Parameters
    ----------
    result:
        Training result returned by ``train_lstm()``.
    config:
        Training pipeline configuration used to produce *result*.
    test_report:
        Evaluation report computed on the held-out test split.
    inference_artifact_path:
        Local path to the exported ``.pt`` inference artefact
        (produced by ``export_inference_artifact()``).
    repo_id:
        Hugging Face repository identifier in ``"username/repo-name"`` form.
    token:
        Hugging Face API token.  Falls back to the ``HF_TOKEN`` environment
        variable when *token* is ``None``.
    private:
        Whether to create the repository as private.  Ignored if the repository
        already exists.
    commit_message:
        Optional git commit message.  Defaults to the model version string.

    Returns
    -------
    str
        Public (or private) URL of the Hugging Face repository.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    ValueError
        If *repo_id* does not contain a ``/`` separator.
    """
    if not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed.\nInstall it with: pip install huggingface_hub"
        )

    if "/" not in repo_id:
        raise ValueError(f"repo_id must be in the form 'username/repo-name', got: {repo_id!r}")

    resolved_token = token or os.environ.get("HF_TOKEN")
    if resolved_token:
        login(token=resolved_token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    model_card_text = _build_model_card(
        result=result, config=config, test_report=test_report, repo_id=repo_id
    )

    is_stgcn = config.model.model_name.startswith("stgcn")

    if is_stgcn:
        model_config_snapshot: dict[str, Any] = {
            "model_name": config.model.model_name,
            "num_nodes": getattr(config.model, "num_nodes", 67),
            "num_coords": getattr(config.model, "num_coords", 3),
            "stgcn_channels": getattr(config.model, "stgcn_channels", [64, 128]),
            "lstm_hidden": getattr(config.model, "lstm_hidden", 128),
            "lstm_layers": getattr(config.model, "lstm_layers", 2),
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
        schema_snapshot: dict[str, Any] = {
            "graph_schema": {
                "graph_schema_version": getattr(
                    config.export, "graph_schema_version", "graph-schema-v1"
                ),
                "num_nodes": getattr(config.model, "num_nodes", 67),
                "num_coords": getattr(config.model, "num_coords", 3),
                "window_length": result.dataset_metadata.get("window_length", 24),
                "window_step": result.dataset_metadata.get("window_step", 3),
            }
        }
    else:
        model_config_snapshot = {
            "model_name": config.model.model_name,
            "input_size": config.model.input_size,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
        schema_snapshot = {
            "feature_schema": {
                "feature_schema_version": config.export.feature_schema_version,
                "feature_dim": config.model.input_size,
                "orientation_normalization": False,
                "window_length": result.dataset_metadata.get("window_length", 24),
                "window_step": result.dataset_metadata.get("window_step", 3),
            }
        }

    config_snapshot: dict[str, Any] = {
        "model_version": result.model_version,
        "model_config": model_config_snapshot,
        **schema_snapshot,
        "training_config": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "grad_clip_norm": config.training.grad_clip_norm,
            "seed": config.training.seed,
            "label_smoothing": 0.1,
        },
        "evaluation": {
            "test_accuracy": test_report.accuracy,
            "test_macro_f1": test_report.macro_f1,
            "test_loss": test_report.loss,
            "calibration_ece": test_report.calibration_ece,
            "per_class_recall": test_report.per_class_recall,
            "per_class_precision": test_report.per_class_precision,
        },
        "class_labels": result.class_labels,
        "created_at": datetime.now(UTC).isoformat(),
    }

    commit_msg = commit_message or f"Upload {result.model_version}"

    # Commit 1: README + config.json
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        readme_path = tmp / "README.md"
        readme_path.write_text(model_card_text, encoding="utf-8")

        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )

    # Commit 2: inference artifact (may be large — streamed separately)
    api.upload_file(
        path_or_fileobj=str(Path(inference_artifact_path)),
        path_in_repo=f"{result.model_version}_inference.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add inference artifact: {result.model_version}",
    )

    return f"https://huggingface.co/{repo_id}"
