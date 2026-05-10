from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from maestro.training.augmentation import AugmentationConfig
from maestro.training.datamodule import DataSplitConfig, GestureDataModule, GraphGestureDataModule
from maestro.training.evaluate import EvaluationReport, evaluate_model
from maestro.utils.reproducibility import seed_everything


@dataclass(slots=True)
class ModelConfig:
    model_name: str
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    num_classes: int
    # STGCN-specific fields (only used when model_name starts with "stgcn").
    num_nodes: int = 67
    num_coords: int = 3
    stgcn_channels: tuple[int, ...] = (64, 128)
    num_stgcn_layers: int = 2
    lstm_hidden: int = 128
    lstm_layers: int = 2
    # Dual-branch STGCN phonological branch fields (only used for "dual_stgcn_*").
    phono_proj_dim: int = 128
    phono_conv_channels: int = 256
    # EnhancedTwoStreamLSTM-specific fields (only used for "two_stream_attn_*").
    num_heads: int = 4
    proj_dim: int = 64


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
    label_smoothing: float = 0.0
    max_samples_per_class: int | None = None
    exclude_classes: list[str] | None = None


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
    feature_schema_version: str
    graph_schema_version: str = "graph-schema-v1"


@dataclass(slots=True)
class TrainingPipelineConfig:
    model: ModelConfig
    training: TrainingConfig
    split: DataSplitConfig
    scheduler: SchedulerConfig
    export: ExportConfig
    augmentation: AugmentationConfig | None = None


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
            num_classes=int(model_data.get("num_classes", 10)),
            # STGCN fields — ignored for two_stream models.
            num_nodes=int(model_data.get("num_nodes", 67)),
            num_coords=int(model_data.get("num_coords", 3)),
            stgcn_channels=tuple(int(c) for c in model_data.get("stgcn_channels", [64, 128])),
            num_stgcn_layers=int(model_data.get("num_stgcn_layers", 2)),
            lstm_hidden=int(model_data.get("lstm_hidden", 128)),
            lstm_layers=int(model_data.get("lstm_layers", 2)),
            phono_proj_dim=int(model_data.get("phono_proj_dim", 128)),
            phono_conv_channels=int(model_data.get("phono_conv_channels", 256)),
            num_heads=int(model_data.get("num_heads", 4)),
            proj_dim=int(model_data.get("proj_dim", 64)),
        ),
        training=TrainingConfig(
            epochs=int(train_data.get("epochs", 30)),
            batch_size=int(train_data.get("batch_size", 64)),
            learning_rate=float(train_data.get("learning_rate", 1e-3)),
            weight_decay=float(train_data.get("weight_decay", 1e-4)),
            grad_clip_norm=float(train_data.get("grad_clip_norm", 1.0)),
            early_stopping_patience=int(train_data.get("early_stopping_patience", 7)),
            early_stopping_min_delta=float(train_data.get("early_stopping_min_delta", 1e-4)),
            seed=int(train_data.get("seed", 42)),
            num_workers=int(train_data.get("num_workers", 0)),
            class_weighting=bool(train_data.get("class_weighting", True)),
            label_smoothing=float(train_data.get("label_smoothing", 0.0)),
            max_samples_per_class=(
                int(train_data["max_samples_per_class"])
                if train_data.get("max_samples_per_class") is not None
                else None
            ),
            exclude_classes=(
                [str(c) for c in train_data["exclude_classes"]]
                if train_data.get("exclude_classes")
                else None
            ),
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
            feature_schema_version=str(
                export_data.get("feature_schema_version", "feature-schema-v1")
            ),
            graph_schema_version=str(
                export_data.get(
                    "graph_schema_version",
                    train_data.get("graph_schema_version", "graph-schema-v1"),
                )
            ),
        ),
        augmentation=_parse_augmentation_config(payload.get("augmentation")),
    )


def _parse_augmentation_config(
    aug_data: dict | None,
) -> AugmentationConfig | None:
    """Return an AugmentationConfig when the YAML section is present and enabled."""
    if not aug_data or not aug_data.get("enabled", False):
        return None

    scale_raw = aug_data.get("temporal_scale_range", [0.8, 1.2])
    return AugmentationConfig(
        temporal_scale_range=(float(scale_raw[0]), float(scale_raw[1])),
        spatial_jitter_sigma=float(aug_data.get("spatial_jitter_sigma", 0.005)),
        apply_prob=float(aug_data.get("apply_prob", 0.5)),
        mirror_flip_prob=float(aug_data.get("mirror_flip_prob", 0.5)),
    )


def _build_model(config: ModelConfig, device: torch.device) -> nn.Module:
    """Instantiate the appropriate model from *config*.

    Dispatches based on ``config.model_name``:
    * ``dual_stgcn_*``      → :class:`DualBranchSTGCN_LSTM`
    * ``two_stream_attn_*`` → :class:`EnhancedTwoStreamLSTM`
    * ``two_stream_*``      → :class:`TwoStreamGestureLSTM`

    Note: ``two_stream_attn_*`` is checked *before* ``two_stream_*`` because it
    is a more specific prefix.
    """
    if config.model_name.startswith("dual_stgcn"):
        from maestro.infrastructure.model.dual_branch_stgcn_lstm import DualBranchSTGCN_LSTM

        return DualBranchSTGCN_LSTM(
            num_nodes=config.num_nodes,
            num_coords=config.num_coords,
            phono_conv_channels=config.phono_conv_channels,
            phono_lstm_hidden=config.lstm_hidden,
            phono_lstm_layers=config.lstm_layers,
            stgcn_channels=config.stgcn_channels,
            dropout=config.dropout,
            num_classes=config.num_classes,
        ).to(device)

    if config.model_name.startswith("two_stream_attn"):
        from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM

        return EnhancedTwoStreamLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            num_heads=config.num_heads,
            proj_dim=config.proj_dim,
        ).to(device)

    if config.model_name.startswith("two_stream"):
        from maestro.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM

        return TwoStreamGestureLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
        ).to(device)

    raise ValueError(
        f"Unsupported model_name {config.model_name!r}. "
        "Supported prefixes: 'dual_stgcn_', 'two_stream_attn_', 'two_stream_'."
    )


def _model_config_from_checkpoint(
    checkpoint_model_config: dict[str, Any] | None,
    fallback: ModelConfig,
    *,
    num_classes: int | None = None,
) -> ModelConfig:
    """Build a ModelConfig from a checkpoint payload with safe fallbacks.

    Fine-tuning must reconstruct the pretrained architecture exactly so the
    saved state dict can be loaded without key or shape drift.  The checkpoint
    model configuration is therefore the source of truth, with *fallback* only
    used for backward-compatible fields that may be absent in older payloads.
    """

    payload = checkpoint_model_config or {}
    resolved_num_classes = (
        int(num_classes)
        if num_classes is not None
        else int(payload.get("num_classes", fallback.num_classes))
    )

    return ModelConfig(
        model_name=str(payload.get("model_name", fallback.model_name)),
        input_size=int(payload.get("input_size", fallback.input_size)),
        hidden_size=int(payload.get("hidden_size", fallback.hidden_size)),
        num_layers=int(payload.get("num_layers", fallback.num_layers)),
        dropout=float(payload.get("dropout", fallback.dropout)),
        num_classes=resolved_num_classes,
        num_nodes=int(payload.get("num_nodes", fallback.num_nodes)),
        num_coords=int(payload.get("num_coords", fallback.num_coords)),
        stgcn_channels=tuple(
            int(c) for c in payload.get("stgcn_channels", fallback.stgcn_channels)
        ),
        num_stgcn_layers=int(payload.get("num_stgcn_layers", fallback.num_stgcn_layers)),
        lstm_hidden=int(payload.get("lstm_hidden", fallback.lstm_hidden)),
        lstm_layers=int(payload.get("lstm_layers", fallback.lstm_layers)),
        phono_proj_dim=int(payload.get("phono_proj_dim", fallback.phono_proj_dim)),
        phono_conv_channels=int(payload.get("phono_conv_channels", fallback.phono_conv_channels)),
        num_heads=int(payload.get("num_heads", fallback.num_heads)),
        proj_dim=int(payload.get("proj_dim", fallback.proj_dim)),
    )


def create_datamodule(
    *,
    dataset_path: str | Path,
    config: TrainingPipelineConfig,
) -> GestureDataModule | GraphGestureDataModule:
    """Create and set up the appropriate data module for *config.model.model_name*.

    * ``dual_stgcn_*`` models → :class:`GraphGestureDataModule`
    * ``two_stream_*``         → :class:`GestureDataModule`
    """
    is_graph_model = config.model.model_name.startswith("dual_stgcn")
    if is_graph_model:
        from maestro.training.augmentation import (  # noqa: F401
            GraphAugmentationConfig,
            GraphGestureAugmenter,
        )

        graph_aug: Any = None
        if config.augmentation is not None:
            graph_aug = GraphAugmentationConfig(
                temporal_scale_range=config.augmentation.temporal_scale_range,
                spatial_jitter_sigma=config.augmentation.spatial_jitter_sigma,
                apply_prob=config.augmentation.apply_prob,
                mirror_flip_prob=config.augmentation.mirror_flip_prob,
            )

        dm: GraphGestureDataModule = GraphGestureDataModule(
            dataset_path=dataset_path,
            batch_size=config.training.batch_size,
            split=config.split,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
            pin_memory=False,
            graph_augmentation=graph_aug,
        )
        dm.setup()
        return dm

    flat_dm: GestureDataModule = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=config.training.batch_size,
        split=config.split,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=False,
        augmentation=config.augmentation,
        max_samples_per_class=config.training.max_samples_per_class,
        exclude_classes=config.training.exclude_classes,
    )
    flat_dm.setup()
    return flat_dm


def _build_criterion(
    class_weights: torch.Tensor | None,
    *,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Construct CrossEntropyLoss with optional class weights and label smoothing."""
    if class_weights is not None:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _build_class_weights(
    datamodule: GestureDataModule | GraphGestureDataModule, device: torch.device
) -> torch.Tensor:
    # Use training-split-only counts so that any per-class downsampling is
    # reflected in the weights.  Without this, the weights would be computed
    # from the full dataset distribution, which still shows the 145x imbalance
    # even after downsampling has been applied to the training split.
    if isinstance(datamodule, GestureDataModule):
        distribution = datamodule.training_label_distribution()
    else:
        distribution = datamodule.label_distribution()
    labels = datamodule.class_labels
    counts = torch.tensor([distribution[label] for label in labels], dtype=torch.float32)
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

    # For flat-feature models, verify input_size matches the dataset.
    if not config.model.model_name.startswith("stgcn") and hasattr(datamodule, "feature_dim"):
        if datamodule.feature_dim != config.model.input_size:  # type: ignore[union-attr]
            raise ValueError(
                "Model input_size does not match dataset feature_dim: "
                f"input_size={config.model.input_size}, feature_dim={datamodule.feature_dim}"  # type: ignore[union-attr]
            )

    if datamodule.num_classes != config.model.num_classes:
        raise ValueError(
            "Model num_classes does not match dataset labels: "
            f"num_classes={config.model.num_classes}, dataset_classes={datamodule.num_classes}"
        )

    model = _build_model(config.model, device)

    if config.training.class_weighting:
        class_weights = _build_class_weights(datamodule, device)
        criterion: nn.Module = _build_criterion(
            class_weights, label_smoothing=config.training.label_smoothing
        )
    else:
        criterion = _build_criterion(None, label_smoothing=config.training.label_smoothing)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler: ReduceLROnPlateau | None = None
    if config.scheduler.enabled:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximise val_macro_f1
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

    best_val_f1 = -float("inf")
    best_val_loss = float("inf")  # tracked for reporting only
    best_epoch = 0
    epochs_without_improve = 0
    history: list[dict[str, float]] = []

    checkpoint_output_dir = Path(output_dir)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_output_dir / f"{model_version}.pt"

    n_train = len(datamodule.train_indices)
    n_val = len(datamodule.val_indices)
    n_params = sum(p.numel() for p in model.parameters())
    _header_cols = (
        f"{'Epoch':>6}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
        f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  "
        f"{'LR':>8}  {'Time':>6}  {'ES':>5}  Note"
    )
    _sep = "-" * len(_header_cols)
    print(f"\nTraining  {model_version}")
    print(f"  Model     : {type(model).__name__}  ({n_params:,} params)")
    print(f"  Dataset   : train={n_train:,}  val={n_val:,}")
    print(
        f"  Optimiser : AdamW  lr={config.training.learning_rate:.2e}  "
        f"wd={config.training.weight_decay:.2e}"
    )
    print(
        f"  Schedule  : ReduceLROnPlateau  "
        f"factor={config.scheduler.factor}  patience={config.scheduler.patience}"
        if config.scheduler.enabled
        else "  Schedule  : none"
    )
    print(
        f"  Stopping  : patience={config.training.early_stopping_patience}  "
        f"min_delta={config.training.early_stopping_min_delta:.0e}  "
        f"metric=val_macro_f1"
    )
    print(_sep)
    print(_header_cols)
    print(_sep, flush=True)

    for epoch in range(1, config.training.epochs + 1):
        _t_epoch = time.perf_counter()
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
            scheduler.step(val_report.macro_f1)

        current_lr = float(optimizer.param_groups[0]["lr"])
        _epoch_secs = time.perf_counter() - _t_epoch
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_report.loss,
                "val_accuracy": val_report.accuracy,
                "val_macro_f1": val_report.macro_f1,
                "learning_rate": current_lr,
                "epoch_secs": _epoch_secs,
            }
        )

        improved = (val_report.macro_f1 - best_val_f1) > config.training.early_stopping_min_delta
        if improved:
            best_val_f1 = val_report.macro_f1
            best_val_loss = val_report.loss  # track for reporting
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
                    "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            _note = "* best"
        else:
            epochs_without_improve += 1
            _note = f"  ES {epochs_without_improve}/{config.training.early_stopping_patience}"

        print(
            f"{epoch:>6}/{config.training.epochs:<4}  "
            f"{train_loss:>9.4f}  {train_acc:>8.1%}  "
            f"{val_report.loss:>8.4f}  {val_report.accuracy:>7.1%}  "
            f"{val_report.macro_f1:>6.1%}  "
            f"{current_lr:>8.2e}  {_epoch_secs:>5.1f}s  "
            f"{_note}",
            flush=True,
        )

        if epochs_without_improve >= config.training.early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch}  "
                f"(no improvement in val_macro_f1 for "
                f"{config.training.early_stopping_patience} epochs)",
                flush=True,
            )
            break

    else:
        print(f"\nCompleted all {config.training.epochs} epochs.", flush=True)

    print(
        f"\nBest epoch  : {best_epoch}  "
        f"val_macro_f1={best_val_f1:.4f}  val_loss={best_val_loss:.4f}",
        flush=True,
    )

    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        "best_val_f1": best_val_f1,
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


def finetune_lstm(
    *,
    dataset_path: str | Path,
    config: TrainingPipelineConfig,
    pretrained_checkpoint_path: str | Path,
    output_dir: str | Path,
    run_root_dir: str | Path,
    device: torch.device,
    freeze_epochs: int = 10,
) -> TrainingResult:
    """Fine-tune a pretrained ``EnhancedTwoStreamLSTM`` on a new dataset.

    Implements a two-stage transfer learning strategy:

    **Stage A — Head-only** (``freeze_epochs`` epochs):
        The pretrained backbone (feature projections, BiLSTMs, self-attention,
        cross-stream gates) is frozen.  Only the new classification head is
        trained with ``lr=1e-3`` so it converges quickly before the backbone
        is unfrozen.

    **Stage B — Full model** (``config.training.epochs`` epochs):
        All layers are unfrozen and jointly fine-tuned with the learning rate
        and schedule specified in *config*.  Early stopping monitors
        ``val_macro_f1`` as in :func:`train_lstm`.

    The pretrained checkpoint must have been saved by :func:`train_lstm` and
    must contain ``model_config`` in its payload so the original architecture
    (``hidden_size``, ``num_heads``, ``proj_dim``, etc.) can be reconstructed
    before loading weights.  The classification head is replaced in-place with
    a new ``Linear`` layer sized for ``config.model.num_classes``.

    Parameters
    ----------
    dataset_path:
        Path to the fine-tuning dataset NPZ (e.g. ``ipn_finetune_dataset.npz``).
    config:
        Fine-tuning pipeline config.  ``config.model.num_classes`` must match
        the number of classes in *dataset_path*.
    pretrained_checkpoint_path:
        Path to the ``.pt`` checkpoint produced by a prior :func:`train_lstm`
        call (e.g. the Jester pretraining checkpoint).
    output_dir:
        Directory where the best fine-tuned checkpoint is saved.
    run_root_dir:
        Directory under which a timestamped run sub-directory is created.
    device:
        Torch device to train on.
    freeze_epochs:
        Number of Stage-A epochs with backbone frozen.  Set to ``0`` to skip
        Stage A and fine-tune all layers from epoch 1.
    """
    seed_everything(config.training.seed)
    datamodule = create_datamodule(dataset_path=dataset_path, config=config)

    if not config.model.model_name.startswith("two_stream_attn"):
        raise ValueError(
            "finetune_lstm only supports 'two_stream_attn_*' models. "
            f"Got: {config.model.model_name!r}"
        )

    # ── Load pretrained checkpoint ────────────────────────────────────────────
    ckpt = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
    pretrained_model_config_dict: dict[str, Any] = ckpt.get("model_config", {})

    # Reconstruct the pretrained model from the checkpoint description so the
    # saved state dict loads against the exact architecture that produced it.
    pretrained_mc = _model_config_from_checkpoint(
        pretrained_model_config_dict,
        config.model,
        num_classes=int(pretrained_model_config_dict.get("num_classes", 27)),
    )
    model = _build_model(pretrained_mc, device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Replace classification head ───────────────────────────────────────────
    # EnhancedTwoStreamLSTM.head is a nn.Sequential ending in Linear(H, old_C).
    # Replace only the final Linear so all other head layers (LN, GELU, Dropout)
    # retain their pretrained weights.
    from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM

    if not isinstance(model, EnhancedTwoStreamLSTM):
        raise TypeError(f"finetune_lstm requires EnhancedTwoStreamLSTM; got {type(model).__name__}")

    if not isinstance(model.head[-1], nn.Linear):
        raise TypeError(
            "EnhancedTwoStreamLSTM.head[-1] must be nn.Linear for fine-tuning head replacement"
        )

    old_head_linear = cast(nn.Linear, model.head[-1])
    new_num_classes = config.model.num_classes
    model.head[-1] = nn.Linear(old_head_linear.in_features, new_num_classes).to(device)
    model.num_classes = new_num_classes

    if datamodule.num_classes != new_num_classes:
        raise ValueError(
            f"config.model.num_classes ({new_num_classes}) does not match "
            f"dataset classes ({datamodule.num_classes})"
        )

    # ── Criterion / scheduler helpers ─────────────────────────────────────────
    def _make_criterion(weights: torch.Tensor | None) -> nn.Module:
        return _build_criterion(weights, label_smoothing=config.training.label_smoothing)

    if config.training.class_weighting:
        class_weights = _build_class_weights(datamodule, device)
        criterion: nn.Module = _make_criterion(class_weights)
    else:
        criterion = _make_criterion(None)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{config.model.model_name}_finetune_{timestamp}"
    run_dir = Path(run_root_dir) / model_version
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_output_dir = Path(output_dir)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_output_dir / f"{model_version}.pt"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nFine-tuning  {model_version}")
    print(f"  Pretrained : {Path(pretrained_checkpoint_path).name}")
    print(f"  Model      : EnhancedTwoStreamLSTM  ({n_params:,} params)")
    print(
        f"  Dataset    : train={len(datamodule.train_indices):,}  "
        f"val={len(datamodule.val_indices):,}"
    )
    print(f"  Classes    : {new_num_classes}  {datamodule.class_labels}")

    history: list[dict[str, float]] = []
    best_val_f1 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Stage A — freeze backbone, train head only
    # ──────────────────────────────────────────────────────────────────────────
    if freeze_epochs > 0:
        # Freeze everything except model.head
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_a = AdamW(head_params, lr=1e-3, weight_decay=config.training.weight_decay)
        _sep = "-" * 80
        print(f"\n{'─' * 80}")
        print(f"Stage A — backbone frozen, head-only  ({freeze_epochs} epochs, lr=1e-3)")
        print(_sep)
        print(
            f"{'Epoch':>6}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
            f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}"
        )
        print(_sep, flush=True)

        for epoch in range(1, freeze_epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = _train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer_a,
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
            history.append(
                {
                    "epoch": float(epoch),
                    "stage": 0.0,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_report.loss,
                    "val_accuracy": val_report.accuracy,
                    "val_macro_f1": val_report.macro_f1,
                    "learning_rate": 1e-3,
                    "epoch_secs": time.perf_counter() - t0,
                }
            )
            print(
                f"{epoch:>6}/{freeze_epochs:<4}  {train_loss:>9.4f}  "
                f"{train_acc:>8.1%}  {val_report.loss:>8.4f}  "
                f"{val_report.accuracy:>7.1%}  {val_report.macro_f1:>6.1%}",
                flush=True,
            )

        print(f"\nStage A complete. Val macro-F1 = {val_report.macro_f1:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Stage B — unfreeze all layers, full fine-tuning
    # ──────────────────────────────────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = True

    optimizer_b = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler: ReduceLROnPlateau | None = None
    if config.scheduler.enabled:
        scheduler = ReduceLROnPlateau(
            optimizer_b,
            mode="max",
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )

    _sep = "-" * 90
    print(f"\n{'─' * 90}")
    print(
        f"Stage B — full fine-tuning  (up to {config.training.epochs} epochs, "
        f"lr={config.training.learning_rate:.2e})"
    )
    print(_sep)
    _header = (
        f"{'Epoch':>6}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
        f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>6}  {'LR':>8}  Note"
    )
    print(_header)
    print(_sep, flush=True)

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer_b,
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
            scheduler.step(val_report.macro_f1)
        current_lr = float(optimizer_b.param_groups[0]["lr"])

        improved = (val_report.macro_f1 - best_val_f1) > config.training.early_stopping_min_delta
        if improved:
            best_val_f1 = val_report.macro_f1
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
                    "pretrained_checkpoint": str(pretrained_checkpoint_path),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            _note = "* best"
        else:
            epochs_without_improve += 1
            _note = f"  ES {epochs_without_improve}/{config.training.early_stopping_patience}"

        history.append(
            {
                "epoch": float(freeze_epochs + epoch),
                "stage": 1.0,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_report.loss,
                "val_accuracy": val_report.accuracy,
                "val_macro_f1": val_report.macro_f1,
                "learning_rate": current_lr,
                "epoch_secs": time.perf_counter() - t0,
            }
        )
        print(
            f"{epoch:>6}/{config.training.epochs:<4}  {train_loss:>9.4f}  "
            f"{train_acc:>8.1%}  {val_report.loss:>8.4f}  {val_report.accuracy:>7.1%}  "
            f"{val_report.macro_f1:>6.1%}  {current_lr:>8.2e}  {_note}",
            flush=True,
        )

        if epochs_without_improve >= config.training.early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch}  "
                f"(no improvement for {config.training.early_stopping_patience} epochs)"
            )
            break
    else:
        print(f"\nCompleted all {config.training.epochs} Stage B epochs.")

    print(
        f"\nBest epoch : {best_epoch}  val_macro_f1={best_val_f1:.4f}  val_loss={best_val_loss:.4f}"
    )

    # ── Load best checkpoint and evaluate ─────────────────────────────────────
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

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

    # ── Persist run artefacts ─────────────────────────────────────────────────
    training_summary = {
        "model_version": model_version,
        "pretrained_checkpoint": str(pretrained_checkpoint_path),
        "freeze_epochs": freeze_epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
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
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )
    (run_dir / "dataset_metadata.json").write_text(
        json.dumps(datamodule.metadata, indent=2), encoding="utf-8"
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(training_summary["config"], sort_keys=False), encoding="utf-8"
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
