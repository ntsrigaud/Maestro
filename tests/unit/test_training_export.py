from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact
from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM
from maestro.training.export_checkpoint import compute_sha256
from maestro.training.train_lstm import (
    DataSplitConfig,
    ExportConfig,
    ModelConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingPipelineConfig,
)

pytestmark = pytest.mark.training


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    digest = compute_sha256(path)
    assert len(digest) == 64


def test_export_inference_artifact_uses_trained_model_shape(tmp_path: Path) -> None:
    from maestro.training.export_checkpoint import export_inference_artifact

    trained_model = EnhancedTwoStreamLSTM(
        input_size=147,
        hidden_size=96,
        num_layers=4,
        dropout=0.35,
        num_classes=14,
        num_heads=8,
        proj_dim=96,
    )
    result = SimpleNamespace(
        model=trained_model,
        model_version="two_stream_attn_finetune_test",
        class_labels=[f"c{i}" for i in range(14)],
        dataset_metadata={
            "feature_dim": 147,
            "window_length": 32,
            "window_step": 1,
            "enable_orientation_normalization": False,
        },
    )
    config = TrainingPipelineConfig(
        model=ModelConfig(
            model_name="two_stream_attn_v1",
            input_size=147,
            hidden_size=96,
            num_layers=2,
            dropout=0.35,
            num_classes=14,
            num_heads=8,
            proj_dim=96,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            seed=42,
            num_workers=0,
            class_weighting=False,
        ),
        split=DataSplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratified=True,
        ),
        scheduler=SchedulerConfig(
            enabled=False,
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
        export=ExportConfig(
            checkpoint_dir="models/checkpoints",
            exported_dir="models/exported",
            feature_schema_version="feature-schema-v5",
        ),
    )

    artifact_path = export_inference_artifact(
        result=result,
        config=config,
        export_dir=tmp_path,
    )

    artifact = load_inference_artifact(
        artifact_path=artifact_path,
        device=torch.device("cpu"),
    )

    assert artifact.model.num_layers == 4
    assert artifact.model.head[-1].out_features == 14

    x = torch.zeros(1, 32, 147)
    with torch.no_grad():
        out = artifact.model(x)
    assert out.shape == (1, 14)
