from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gesture_control.training.datamodule import DataSplitConfig, GestureDataModule
from gesture_control.training.evaluate import EvaluationReport, evaluate_model
from gesture_control.utils.reproducibility import seed_everything


@dataclass(slots=True)
class ModelConfig:
    model_name: str
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    num_classes: int


@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    seed: int
    num_workers: int
    class_weighting: bool


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool
    factor: float
    patience: int
    min_lr: float


@dataclass(slots=True)
class ExportConfig:
    checkpoint_dir: str
    exported_dir: str
    registry_path: str
    feature_schema_version: str


@dataclass(slots=True)
class TrainingPipelineConfig:
    model: ModelConfig
    training: TrainingConfig
    split: DataSplitConfig
    scheduler: SchedulerConfig
    export: ExportConfig


@dataclass(slots=True)
class TrainingResult:
    model: nn.Module
    model_version: str
    checkpoint_path: Path
    run_dir: Path
    train_history: list[dict[str, float]]
    val_report: EvaluationReport
    test_report: EvaluationReport
    class_labels: list[str]
    dataset_metadata: dict[str, Any]


class GestureLSTMClassifier(nn.Module):
    """Baseline 2-layer LSTM classifier for sequence gesture recognition."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        final_state = output[:, -1, :]
        return self.head(self.dropout(final_state))


def load_training_config(config_path: str | Path) -> TrainingPipelineConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {path}")

    if "model" in payload:
        model_data = payload.get("model", {})
        train_data = payload.get("training", {})
        split_data = payload.get("splits", {})
        scheduler_data = payload.get("scheduler", {})
        export_data = payload.get("export", {})
    else:
        # Backward-compatible flat config parsing.
        model_data = payload
        train_data = {}
        split_data = {}
        scheduler_data = {}
        export_data = {}

    return TrainingPipelineConfig(
        model=ModelConfig(
            model_name=str(model_data.get("model_name", "lstm_v1")),
            input_size=int(model_data.get("input_size", 126)),
            hidden_size=int(model_data.get("hidden_size", 128)),
            num_layers=int(model_data.get("num_layers", 2)),
            dropout=float(model_data.get("dropout", 0.3)),
            num_classes=int(model_data.get("num_classes", 8)),
        ),
        training=TrainingConfig(
            epochs=int(train_data.get("epochs", 30)),
            batch_size=int(train_data.get("batch_size", 64)),
            learning_rate=float(train_data.get("learning_rate", 1e-3)),
            weight_decay=float(train_data.get("weight_decay", 1e-4)),
            grad_clip_norm=float(train_data.get("grad_clip_norm", 1.0)),
            early_stopping_patience=int(train_data.get("early_stopping_patience", 7)),
            early_stopping_min_delta=float(
                train_data.get("early_stopping_min_delta", 1e-4)
            ),
            seed=int(train_data.get("seed", 42)),
            num_workers=int(train_data.get("num_workers", 0)),
            class_weighting=bool(train_data.get("class_weighting", True)),
        ),
        split=DataSplitConfig(
            train_ratio=float(split_data.get("train_ratio", 0.7)),
            val_ratio=float(split_data.get("val_ratio", 0.15)),
            test_ratio=float(split_data.get("test_ratio", 0.15)),
            stratified=bool(split_data.get("stratified", True)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler_data.get("enabled", True)),
            factor=float(scheduler_data.get("factor", 0.5)),
            patience=int(scheduler_data.get("patience", 3)),
            min_lr=float(scheduler_data.get("min_lr", 1e-6)),
        ),
        export=ExportConfig(
            checkpoint_dir=str(export_data.get("checkpoint_dir", "models/checkpoints")),
            exported_dir=str(export_data.get("exported_dir", "models/exported")),
            registry_path=str(export_data.get("registry_path", "models/registry.json")),
            feature_schema_version=str(
                export_data.get("feature_schema_version", "feature-schema-v1")
            ),
        ),
    )


def create_datamodule(
    *,
    dataset_path: str | Path,
    config: TrainingPipelineConfig,
) -> GestureDataModule:
    datamodule = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=config.training.batch_size,
        split=config.split,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=False,
    )
    datamodule.setup()
    return datamodule


def _build_class_weights(
    datamodule: GestureDataModule, device: torch.device
) -> torch.Tensor:
    distribution = datamodule.label_distribution()
    labels = datamodule.class_labels
    counts = torch.tensor(
        [distribution[label] for label in labels], dtype=torch.float32
    )
    weights = torch.sum(counts) / (counts * len(labels))
    return weights.to(device)


def _train_one_epoch(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * batch_x.size(0)
        total_correct += int((preds == batch_y).sum().item())
        total_examples += int(batch_x.size(0))

    if total_examples == 0:
        raise RuntimeError("Train dataloader produced no samples")

    return float(total_loss / total_examples), float(total_correct / total_examples)


def train_lstm(
    *,
    dataset_path: str | Path,
    config: TrainingPipelineConfig,
    output_dir: str | Path,
    run_root_dir: str | Path,
    device: torch.device,
) -> TrainingResult:
    seed_everything(config.training.seed)
    datamodule = create_datamodule(dataset_path=dataset_path, config=config)

    if datamodule.feature_dim != config.model.input_size:
        raise ValueError(
            "Model input_size does not match dataset feature_dim: "
            f"input_size={config.model.input_size}, feature_dim={datamodule.feature_dim}"
        )

    if datamodule.num_classes != config.model.num_classes:
        raise ValueError(
            "Model num_classes does not match dataset labels: "
            f"num_classes={config.model.num_classes}, dataset_classes={datamodule.num_classes}"
        )

    model = GestureLSTMClassifier(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes,
    ).to(device)

    if config.training.class_weighting:
        class_weights = _build_class_weights(datamodule, device)
        criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler: ReduceLROnPlateau | None = None
    if config.scheduler.enabled:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{config.model.model_name}_{timestamp}"

    run_dir = Path(run_root_dir) / model_version
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    history: list[dict[str, float]] = []

    checkpoint_output_dir = Path(output_dir)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_output_dir / f"{model_version}.pt"

    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=config.training.grad_clip_norm,
        )

        val_report = evaluate_model(
            model=model,
            dataloader=val_loader,
            class_labels=datamodule.class_labels,
            device=device,
            criterion=criterion,
        )

        if scheduler is not None:
            scheduler.step(val_report.loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_report.loss,
                "val_accuracy": val_report.accuracy,
                "val_macro_f1": val_report.macro_f1,
                "learning_rate": current_lr,
            }
        )

        improved = (
            best_val_loss - val_report.loss
        ) > config.training.early_stopping_min_delta
        if improved:
            best_val_loss = val_report.loss
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config.model),
                    "class_labels": datamodule.class_labels,
                    "dataset_metadata": datamodule.metadata,
                    "feature_schema_version": config.export.feature_schema_version,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= config.training.early_stopping_patience:
            break

    checkpoint_payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    val_report = evaluate_model(
        model=model,
        dataloader=val_loader,
        class_labels=datamodule.class_labels,
        device=device,
        criterion=criterion,
    )
    test_report = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_labels=datamodule.class_labels,
        device=device,
        criterion=criterion,
    )

    training_summary = {
        "model_version": model_version,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "val_report": val_report.to_dict(),
        "test_report": test_report.to_dict(),
        "config": {
            "model": asdict(config.model),
            "training": asdict(config.training),
            "split": asdict(config.split),
            "scheduler": asdict(config.scheduler),
            "export": asdict(config.export),
        },
        "dataset_metadata": datamodule.metadata,
        "label_distribution": datamodule.label_distribution(),
    }
    (run_dir / "training_metrics.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "dataset_metadata.json").write_text(
        json.dumps(datamodule.metadata, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(training_summary["config"], sort_keys=False),
        encoding="utf-8",
    )

    return TrainingResult(
        model=model,
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        train_history=history,
        val_report=val_report,
        test_report=test_report,
        class_labels=datamodule.class_labels,
        dataset_metadata=datamodule.metadata,
    )
