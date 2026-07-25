from __future__ import annotations

import copy
import itertools
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from maestro.training.augmentation import AugmentationConfig
from maestro.training.datamodule import DataSplitConfig, GestureDataModule
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
    # EnhancedTwoStreamLSTM-specific fields (only used for "two_stream_attn_*").
    num_heads: int = 4
    proj_dim: int = 64
    motion_temporal_conv: bool = False
    motion_attn_pool: bool = False
    # IPN baseline adapter-specific fields (only used for "ipn_baseline_*").
    baseline_backbone: str = "resnet18"
    sample_size: int = 32
    width_mult: float = 1.0
    shortcut_type: str = "B"
    cardinality: int = 32


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
            num_heads=int(model_data.get("num_heads", 4)),
            proj_dim=int(model_data.get("proj_dim", 64)),
            motion_temporal_conv=bool(model_data.get("motion_temporal_conv", False)),
            motion_attn_pool=bool(model_data.get("motion_attn_pool", False)),
            baseline_backbone=str(model_data.get("baseline_backbone", "resnet18")),
            sample_size=int(model_data.get("sample_size", 32)),
            width_mult=float(model_data.get("width_mult", 1.0)),
            shortcut_type=str(model_data.get("shortcut_type", "B")),
            cardinality=int(model_data.get("cardinality", 32)),
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


def _build_model(
    config: ModelConfig,
    device: torch.device,
    *,
    sequence_length: int | None = None,
) -> nn.Module:
    """Instantiate the appropriate model from *config*.

    Dispatches based on ``config.model_name``:
    * ``two_stream_attn_*`` → :class:`EnhancedTwoStreamLSTM`
    * ``ipn_baseline_*``   → :class:`IPNFeatureToVideoAdapter`
    """
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
            motion_temporal_conv=config.motion_temporal_conv,
            motion_attn_pool=config.motion_attn_pool,
        ).to(device)

    if config.model_name.startswith("ipn_baseline"):
        if sequence_length is None:
            raise ValueError("sequence_length is required for ipn_baseline_* models")
        from maestro.infrastructure.model.ipn_baseline_adapter import (
            IPNBaselineAdapterConfig,
            IPNFeatureToVideoAdapter,
        )

        adapter_cfg = IPNBaselineAdapterConfig(
            input_size=config.input_size,
            num_classes=config.num_classes,
            sequence_length=sequence_length,
            sample_size=config.sample_size,
            width_mult=config.width_mult,
            shortcut_type=config.shortcut_type,
            cardinality=config.cardinality,
        )
        return IPNFeatureToVideoAdapter(
            backbone_name=config.baseline_backbone,
            config=adapter_cfg,
        ).to(device)

    raise ValueError(
        f"Unsupported model_name {config.model_name!r}. "
        "Supported prefixes: 'two_stream_attn_', 'ipn_baseline_'."
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
        num_heads=int(payload.get("num_heads", fallback.num_heads)),
        proj_dim=int(payload.get("proj_dim", fallback.proj_dim)),
        # Default to False — old checkpoints were trained without these flags.
        # The variant architecture is applied below in finetune_lstm after
        # the pretrained weights are loaded into the baseline model.
        motion_temporal_conv=bool(payload.get("motion_temporal_conv", False)),
        motion_attn_pool=bool(payload.get("motion_attn_pool", False)),
    )


def create_datamodule(
    *,
    dataset_path: str | Path,
    config: TrainingPipelineConfig,
) -> GestureDataModule:
    """Create and set up the feature-based GestureDataModule."""
    dm = GestureDataModule(
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
    dm.setup()
    return dm


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


def _build_class_weights(datamodule: GestureDataModule, device: torch.device) -> torch.Tensor:
    # Use training-split-only counts so that any per-class downsampling is
    # reflected in the weights.  Without this, the weights would be computed
    # from the full dataset distribution, which still shows the 145x imbalance
    # even after downsampling has been applied to the training split.
    distribution = datamodule.training_label_distribution()
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


def _train_one_epoch_with_retention(
    *,
    model: nn.Module,
    teacher: nn.Module,
    target_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    replay_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    replay_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    target_criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
    distillation_weight: float,
    replay_ce_weight: float,
    jester_subspace: dict[str, torch.Tensor] | None = None,
) -> tuple[float, float, float, float]:
    model.train()
    teacher.eval()

    total_loss = 0.0
    total_distill_loss = 0.0
    total_replay_ce_loss = 0.0
    total_correct = 0
    total_examples = 0

    student_with_features = cast(Any, model)
    teacher_with_features = cast(Any, teacher)

    for batch_x, batch_y in target_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        replay_x, replay_y = next(replay_iter)
        replay_x = replay_x.to(device)
        replay_y = replay_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        target_loss = target_criterion(logits, batch_y)

        student_features = student_with_features.forward_features(replay_x)
        with torch.no_grad():
            teacher_features = teacher_with_features.forward_features(replay_x)

        distill_loss = F.mse_loss(student_features, teacher_features)
        replay_logits = replay_head(student_features)
        replay_ce_loss = F.cross_entropy(replay_logits, replay_y)

        loss = (
            target_loss + (distillation_weight * distill_loss) + (replay_ce_weight * replay_ce_loss)
        )
        loss.backward()

        # GPM: project weight gradients to null-space of Jester activation subspace
        if jester_subspace is not None:
            _project_gradients_gpm(model, jester_subspace)

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * batch_x.size(0)
        total_distill_loss += float(distill_loss.item()) * batch_x.size(0)
        total_replay_ce_loss += float(replay_ce_loss.item()) * batch_x.size(0)
        total_correct += int((preds == batch_y).sum().item())
        total_examples += int(batch_x.size(0))

    if total_examples == 0:
        raise RuntimeError("Train dataloader produced no samples")

    return (
        float(total_loss / total_examples),
        float(total_correct / total_examples),
        float(total_distill_loss / total_examples),
        float(total_replay_ce_loss / total_examples),
    )


def _evaluate_jester_f1(
    *,
    model: nn.Module,
    replay_head: nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """Return Jester macro-F1 via backbone features → frozen Jester head.

    Used during Stage B to measure old-task retention without requiring the
    Jester head to be attached to the live model (which has an IPN head after
    head replacement).  Computes per-class F1 and returns the unweighted macro
    average.
    """
    model.eval()
    replay_head.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in dataloader:
            features = cast(Any, model).forward_features(x.to(device))
            preds = torch.argmax(replay_head(features), dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y.cpu())
    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    classes = labels_cat.unique().tolist()
    per_class_f1: list[float] = []
    for c in classes:
        c_int = int(c)
        tp = int(((preds_cat == c_int) & (labels_cat == c_int)).sum())
        fp = int(((preds_cat == c_int) & (labels_cat != c_int)).sum())
        fn = int(((preds_cat != c_int) & (labels_cat == c_int)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
        per_class_f1.append(f1)
    return float(sum(per_class_f1) / len(per_class_f1)) if per_class_f1 else 0.0


def _compute_jester_subspace(
    *,
    model: nn.Module,
    replay_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_components: int = 20,
) -> dict[str, torch.Tensor]:
    """Compute top-``num_components`` principal directions of Jester input activations.

    Registers forward hooks on non-head Linear modules to collect **input**
    activations over the replay set, then runs truncated SVD to extract the
    principal component directions in input space.

    Returns a dict mapping module name → ``(in_features, k)`` matrix whose
    columns are the top-k right singular vectors of the centred input activation
    matrix.  These directions are used by :func:`_project_gradients_gpm` to
    constrain Stage B weight gradients to the orthogonal complement of the
    Jester input subspace.

    Per Saha et al. 2020 (GPM), the subspace is computed from **input**
    activations so that the projection operates in the same space as the weight
    columns (``W`` has shape ``(out_features, in_features)``).
    """
    activations: dict[str, list[torch.Tensor]] = {}
    hooks: list[Any] = []

    def _make_hook(name: str) -> Any:
        def _hook(module: nn.Module, inp: Any, out: Any) -> None:
            x = inp[0]  # (batch, *, in_features) — first positional input
            # Collapse all leading dims except the last (in_features) so each
            # timestep / position contributes an independent row to the SVD.
            activations.setdefault(name, []).append(
                x.detach().cpu().reshape(-1, x.shape[-1])  # (N, in_features)
            )

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.startswith("head"):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for x, _ in replay_loader:
            model(x.to(device))

    for h in hooks:
        h.remove()

    subspace: dict[str, torch.Tensor] = {}
    for name, acts in activations.items():
        A = torch.cat(acts, dim=0).float()  # (N, in_features)
        k = min(num_components, A.shape[0] - 1, A.shape[1])
        if k < 1:
            continue
        A = A - A.mean(dim=0, keepdim=True)  # centre activations
        _, _, Vt = torch.linalg.svd(A, full_matrices=False)
        subspace[name] = Vt[:k].T  # (in_features, k) — principal directions as columns
    return subspace


def _project_gradients_gpm(
    model: nn.Module,
    subspace: dict[str, torch.Tensor],
) -> None:
    """Project weight gradients to the null-space of the Jester input activation subspace.

    For each Linear module present in ``subspace``, the weight gradient matrix
    is projected to the orthogonal complement of the module's Jester principal
    input directions::

        dW = dW - (dW @ U) @ U.T

    where ``U`` is the ``(in_features, k)`` matrix of Jester principal input
    components (columns).  This operates row-wise in input space: each output
    neuron's incoming weight vector is projected away from the Jester input
    directions, preserving the directions that encode Jester temporal gesture
    patterns (GPM, Saha et al. 2020).

    Gradients are modified **in-place** and should be called immediately after
    ``loss.backward()`` and before ``clip_grad_norm_`` and the optimiser step.
    """
    for name, module in model.named_modules():
        if name not in subspace:
            continue
        U = subspace[name]  # (in_features, k) — orthonormal columns
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            continue
        U_dev = U.to(weight.grad.device)
        # weight.grad shape: (out_features, in_features)
        # Project each output neuron's gradient away from Jester input subspace:
        #   dW = dW - (dW @ U) @ U.T
        dW = weight.grad  # (out_features, in_features)
        weight.grad.copy_(dW - (dW @ U_dev) @ U_dev.T)


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

    model = _build_model(config.model, device, sequence_length=datamodule.sequence_length)

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
    stage_b_warmup_epochs: int = 5,
    replay_dataset_path: str | Path | None = None,
    replay_max_samples_per_class: int = 500,
    distillation_weight: float = 0.5,
    replay_ce_weight: float = 0.3,
    backbone_lr_multiplier: float = 0.1,
    forgetting_penalty_weight: float = 0.5,
    gpm_components: int = 20,
) -> TrainingResult:
    """Fine-tune a pretrained ``EnhancedTwoStreamLSTM`` on a new dataset.

    Implements a two-stage transfer learning strategy:

    **Stage A — Head-only** (``freeze_epochs`` epochs):
        The pretrained backbone (feature projections, BiLSTMs, self-attention,
        cross-stream gates) is frozen. Only the new classification head is
        trained with ``lr=1e-3`` so it converges quickly before the backbone
        is unfrozen.

    **Stage B — Full model** (``config.training.epochs`` epochs):
        All layers are unfrozen and jointly fine-tuned with:
        - Backbone LR linear warmup over ``stage_b_warmup_epochs`` epochs
          (ramps from 0 → target to prevent the loss spike caused by the
          cold AdamW state of previously-frozen backbone parameters).
        - Feature-level distillation (MSE between student/teacher features)
        - Frozen-head replay CE (old-task classification objective)
        - GPM gradient projection (Jester subspace orthogonality constraint)
        - Jester retention monitoring (macro-F1 on held-out Jester val split)
        Early stopping uses a composite score that penalises Jester forgetting.
        The ReduceLROnPlateau scheduler patience counter only starts after
        the backbone warmup completes.
    """
    seed_everything(config.training.seed)
    datamodule = create_datamodule(dataset_path=dataset_path, config=config)

    if not config.model.model_name.startswith("two_stream_attn"):
        raise ValueError(
            "finetune_lstm only supports 'two_stream_attn_*' models. "
            f"Got: {config.model.model_name!r}"
        )

    ckpt = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
    pretrained_model_config_dict: dict[str, Any] = ckpt.get("model_config", {})

    pretrained_mc = _model_config_from_checkpoint(
        pretrained_model_config_dict,
        config.model,
        num_classes=int(pretrained_model_config_dict.get("num_classes", 27)),
    )
    from dataclasses import replace as _dc_replace

    # Allow depth adaptation (e.g., 4-layer pretrained -> 2-layer fine-tune)
    # while keeping all other checkpoint-defined backbone dimensions intact.
    target_num_layers = int(config.model.num_layers)
    loaded_mc = (
        _dc_replace(pretrained_mc, num_layers=target_num_layers)
        if target_num_layers != pretrained_mc.num_layers
        else pretrained_mc
    )

    model = _build_model(loaded_mc, device)
    if target_num_layers == pretrained_mc.num_layers:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        missing = len(incompatible.missing_keys)
        unexpected = len(incompatible.unexpected_keys)
        print(
            "  Depth adapt : "
            f"checkpoint num_layers={pretrained_mc.num_layers} -> "
            f"target num_layers={target_num_layers}"
        )
        print(f"  Partial load: missing_keys={missing}, unexpected_keys={unexpected}")

    # ── Apply motion-branch variant flags if they differ from the pretrained checkpoint ──
    # Old checkpoints don't store these flags (they default to False above).  If the
    # target config requests a variant, build a new model with those flags and copy
    # all matching weights (strict=False).  New layers (tcn_b, attn_pool_b) are
    # randomly initialised — they will be trained from scratch during fine-tuning.
    if (
        config.model.motion_temporal_conv != loaded_mc.motion_temporal_conv
        or config.model.motion_attn_pool != loaded_mc.motion_attn_pool
    ):
        _variant_mc = _dc_replace(
            loaded_mc,
            motion_temporal_conv=config.model.motion_temporal_conv,
            motion_attn_pool=config.model.motion_attn_pool,
        )
        _variant_model = _build_model(_variant_mc, device)
        _incompatible = _variant_model.load_state_dict(model.state_dict(), strict=False)
        if _incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when building variant model: {_incompatible.unexpected_keys}"
            )
        model = _variant_model
        del _variant_mc, _variant_model, _incompatible

    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Keep a frozen copy of the source head (27-class by default) and apply it to
    # current backbone features on replay batches. This creates an explicit
    # old-task classification retention objective during Stage B.
    replay_head = copy.deepcopy(cast(nn.Module, model.head)).to(device)
    replay_head.eval()
    for param in replay_head.parameters():
        param.requires_grad = False

    # ── Jester replay DM: created here (before Stage A) so we can measure the
    # Jester retention baseline before any IPN training begins.
    replay_dm: GestureDataModule | None = None
    jester_eval_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    jester_pretrain_f1 = float("nan")
    if replay_dataset_path is not None:
        replay_dm = GestureDataModule(
            dataset_path=replay_dataset_path,
            batch_size=config.training.batch_size,
            split=config.split,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
            pin_memory=False,
            augmentation=None,
            max_samples_per_class=replay_max_samples_per_class,
            exclude_classes=None,
        )
        replay_dm.setup()
        jester_eval_loader = replay_dm.val_dataloader()
        # Jester retention BEFORE any IPN training: use teacher model (unmodified backbone)
        jester_pretrain_f1 = _evaluate_jester_f1(
            model=teacher_model,
            replay_head=replay_head,
            dataloader=jester_eval_loader,
            device=device,
        )

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
    if replay_dataset_path is not None:
        print(
            "  Phase 1    : "
            f"replay={Path(replay_dataset_path).name}  "
            f"replay_max/class={replay_max_samples_per_class}  "
            f"distill_w={distillation_weight:.3f}  "
            f"replay_ce_w={replay_ce_weight:.3f}  "
            f"backbone_lr_mult={backbone_lr_multiplier:.3f}  "
            f"gpm_k={gpm_components}  "
            f"forget_pen={forgetting_penalty_weight:.2f}"
        )
        print(f"  Jester baseline val macro-F1 = {jester_pretrain_f1:.4f}")

    history: list[dict[str, float]] = []
    best_val_f1 = -float("inf")
    best_composite = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    if freeze_epochs > 0:
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

    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [
        param for name, param in model.named_parameters() if not name.startswith("head.")
    ]
    head_params = [param for name, param in model.named_parameters() if name.startswith("head.")]
    target_backbone_lr = config.training.learning_rate * backbone_lr_multiplier
    # Start backbone at 0 when warmup is enabled; the training loop ramps it
    # linearly to target_backbone_lr over stage_b_warmup_epochs epochs.  This
    # prevents the loss spike caused by AdamW's cold (zero) second-moment state
    # for backbone params that were frozen throughout Stage A.
    initial_backbone_lr = 0.0 if stage_b_warmup_epochs > 0 else target_backbone_lr
    optimizer_b = AdamW(
        [
            {
                "params": backbone_params,
                "lr": initial_backbone_lr,
            },
            {
                "params": head_params,
                "lr": config.training.learning_rate,
            },
        ],
        weight_decay=config.training.weight_decay,
    )

    # Create replay iterator from the training split of the replay DM created earlier.
    # The val split of replay_dm is held out exclusively for Jester retention monitoring.
    replay_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None
    if replay_dm is not None:
        replay_iter = itertools.cycle(iter(replay_dm.train_dataloader()))

    # ── GPM: Jester activation subspace for gradient projection ───────────────
    # Computed on the teacher backbone so principal directions reflect Jester's
    # feature geometry. GPM projects each Stage B gradient update to the null-
    # space of these directions, preventing IPN from overwriting Jester weights.
    jester_subspace: dict[str, torch.Tensor] | None = None
    if gpm_components > 0 and replay_dm is not None:
        print(f"Computing Jester activation subspace (k={gpm_components}) for GPM...")
        jester_subspace = _compute_jester_subspace(
            model=teacher_model,
            replay_loader=replay_dm.train_dataloader(),
            device=device,
            num_components=gpm_components,
        )
        print(f"  Subspace computed for {len(jester_subspace)} layers.")

    scheduler: ReduceLROnPlateau | None = None
    if config.scheduler.enabled:
        scheduler = ReduceLROnPlateau(
            optimizer_b,
            mode="max",
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )

    _sep = "-" * 110
    print(f"\n{'─' * 110}")
    print(
        f"Stage B — full fine-tuning  (up to {config.training.epochs} epochs, "
        f"head_lr={config.training.learning_rate:.2e}, "
        f"backbone_lr={target_backbone_lr:.2e}  "
        f"[warmup {stage_b_warmup_epochs} epochs: 0→target])"
    )
    if jester_subspace is not None:
        print(f"  GPM projection      {len(jester_subspace)} layers  k={gpm_components}")
    if jester_eval_loader is not None:
        print(
            f"  Jester monitoring   baseline_f1={jester_pretrain_f1:.4f}  "
            f"forgetting_penalty_weight={forgetting_penalty_weight:.2f}"
        )
    print(_sep)
    print(
        f"{'Epoch':>6}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>8}  "
        f"{'ValAcc':>7}  {'ValF1':>6}  {'JstrF1':>7}  {'LR':>8}  "
        f"{'Distill':>8}  {'ReplayCE':>8}  Note"
    )
    print(_sep, flush=True)

    for epoch in range(1, config.training.epochs + 1):
        # ── Backbone LR linear warmup ──────────────────────────────────────────
        # Ramp backbone LR from 0 → target_backbone_lr over the first
        # stage_b_warmup_epochs epochs.  This prevents the loss spike that
        # occurs because AdamW's second-moment estimate (v_t) starts at zero
        # for backbone params that were frozen throughout Stage A — without
        # warmup the initial adaptive step is unreasonably large.
        if stage_b_warmup_epochs > 0 and epoch <= stage_b_warmup_epochs:
            warmup_factor = epoch / stage_b_warmup_epochs
            optimizer_b.param_groups[0]["lr"] = target_backbone_lr * warmup_factor

        t0 = time.perf_counter()
        distill_loss_value = float("nan")
        replay_ce_loss_value = float("nan")
        if replay_iter is not None:
            train_loss, train_acc, distill_loss_value, replay_ce_loss_value = (
                _train_one_epoch_with_retention(
                    model=model,
                    teacher=teacher_model,
                    target_dataloader=train_loader,
                    replay_iter=replay_iter,
                    replay_head=replay_head,
                    optimizer=optimizer_b,
                    target_criterion=criterion,
                    device=device,
                    grad_clip_norm=config.training.grad_clip_norm,
                    distillation_weight=distillation_weight,
                    replay_ce_weight=replay_ce_weight,
                    jester_subspace=jester_subspace,
                )
            )
        else:
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

        # ── Jester retention monitoring ────────────────────────────────────────
        jester_val_f1 = float("nan")
        if jester_eval_loader is not None:
            jester_val_f1 = _evaluate_jester_f1(
                model=model,
                replay_head=replay_head,
                dataloader=jester_eval_loader,
                device=device,
            )

        # Only step the LR scheduler after backbone warmup has completed so
        # that the ReduceLROnPlateau patience counter is not wasted on warmup
        # epochs where loss is artificially high due to the ramp.
        if scheduler is not None and epoch > stage_b_warmup_epochs:
            scheduler.step(val_report.macro_f1)
        current_lr = float(optimizer_b.param_groups[-1]["lr"])
        current_backbone_lr = float(optimizer_b.param_groups[0]["lr"])

        # ── Composite early stopping score ────────────────────────────────────
        # Penalise the IPN val F1 by the amount of Jester forgetting so that
        # the best checkpoint balances IPN performance with source-task retention.
        jester_gap = 0.0
        if jester_eval_loader is not None and jester_pretrain_f1 == jester_pretrain_f1:  # not nan
            jester_gap = max(0.0, jester_pretrain_f1 - jester_val_f1)
        composite = val_report.macro_f1 - forgetting_penalty_weight * jester_gap

        improved = (composite - best_composite) > config.training.early_stopping_min_delta
        if improved:
            best_composite = composite
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
                    "best_composite": best_composite,
                    "jester_val_f1": jester_val_f1,
                },
                checkpoint_path,
            )
            note = "* best"
        else:
            epochs_without_improve += 1
            note = f"  ES {epochs_without_improve}/{config.training.early_stopping_patience}"

        history.append(
            {
                "epoch": float(freeze_epochs + epoch),
                "stage": 1.0,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_report.loss,
                "val_accuracy": val_report.accuracy,
                "val_macro_f1": val_report.macro_f1,
                "jester_val_f1": jester_val_f1,
                "composite_score": composite,
                "learning_rate": current_lr,
                "backbone_lr": current_backbone_lr,
                "stage_b_warmup": epoch <= stage_b_warmup_epochs,
                "epoch_secs": time.perf_counter() - t0,
                "distill_loss": distill_loss_value,
                "replay_ce_loss": replay_ce_loss_value,
            }
        )
        print(
            f"{epoch:>6}/{config.training.epochs:<4}  {train_loss:>9.4f}  {train_acc:>8.1%}  "
            f"{val_report.loss:>8.4f}  {val_report.accuracy:>7.1%}  {val_report.macro_f1:>6.1%}  "
            f"{jester_val_f1:>7.4f}  "
            f"{current_lr:>8.2e}  {distill_loss_value:>8.4f}  "
            f"{replay_ce_loss_value:>8.4f}  {note}",
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
        f"\nBest epoch : {best_epoch}  val_macro_f1={best_val_f1:.4f}  "
        f"composite={best_composite:.4f}  val_loss={best_val_loss:.4f}"
    )

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

    training_summary = {
        "model_version": model_version,
        "pretrained_checkpoint": str(pretrained_checkpoint_path),
        "freeze_epochs": freeze_epochs,
        "stage_b_warmup_epochs": stage_b_warmup_epochs,
        "replay_dataset_path": (
            str(replay_dataset_path) if replay_dataset_path is not None else None
        ),
        "replay_max_samples_per_class": replay_max_samples_per_class,
        "distillation_weight": distillation_weight,
        "replay_ce_weight": replay_ce_weight,
        "backbone_lr_multiplier": backbone_lr_multiplier,
        "forgetting_penalty_weight": forgetting_penalty_weight,
        "gpm_components": gpm_components,
        "jester_pretrain_f1": jester_pretrain_f1,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "best_composite": best_composite,
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
