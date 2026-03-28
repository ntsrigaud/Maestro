from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@dataclass(slots=True)
class EvaluationReport:
    loss: float
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: list[list[int]]
    calibration_ece: float
    mean_confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if confidences.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        ece += (np.sum(mask) / confidences.size) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_model(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    class_labels: list[str],
    device: torch.device,
    criterion: torch.nn.Module,
) -> EvaluationReport:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_confs: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            total_loss += float(loss.item()) * batch_x.size(0)
            total_examples += int(batch_x.size(0))

            all_targets.extend(batch_y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    if total_examples == 0:
        raise RuntimeError("Evaluation dataloader produced no samples")

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_conf = np.asarray(all_confs, dtype=np.float32)

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_labels)),
        zero_division=0,
    )
    correct = (y_true == y_pred).astype(np.float32)

    return EvaluationReport(
        loss=float(total_loss / total_examples),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(class_labels)),
                average="macro",
                zero_division=0,
            )
        ),
        per_class_precision={
            class_labels[idx]: float(value) for idx, value in enumerate(precision)
        },
        per_class_recall={
            class_labels[idx]: float(value) for idx, value in enumerate(recall)
        },
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(class_labels)),
        )
        .astype(int)
        .tolist(),
        calibration_ece=expected_calibration_error(y_conf, correct, n_bins=10),
        mean_confidence=float(np.mean(y_conf)) if y_conf.size > 0 else 0.0,
    )
