#!/usr/bin/env python3
"""
Static Gesture Evaluation — Top-K Accuracy, ECE, Confusion Matrix
==================================================================
Evaluate the Architecture A dual-stream supervised gesture recognition
system on the static HaGRID 8-class subset.

Metrics:
    Top-1 / Top-3 / Top-5 accuracy
    Per-class accuracy breakdown
    Confusion matrix
    Confidence margin distribution

Usage:
    python -m semantic_hgr.scripts.static_eval \\
        --dataset data/hagrid_subset \\
        --checkpoint checkpoints/dual/best.pt \\
        --output results/static_eval_report.json \\
        --top-k 1,3,5

Dataset layout:
    data/hagrid_subset/
        call_gesture/    *.jpg
        closed_fist/     *.jpg
        ok_sign/         *.jpg
        open_palm/       *.jpg
        peace_sign/      *.jpg
        pointing/        *.jpg
        thumbs_down/     *.jpg
        thumbs_up/       *.jpg

Output:
    JSON report containing top-K accuracy, per-class breakdown,
    confusion matrix, and margin distribution statistics.

Note:
    DualStreamModel is a stub until Architecture A training is complete.
    The evaluation framework (dataset loading, metrics) is fully functional
    and can be used to validate the pipeline before training.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import cv2

# Architecture A — 8 supervised gesture classes (HaGRID subset)
GESTURE_CLASSES: List[str] = [
    "call_gesture",
    "closed_fist",
    "ok_sign",
    "open_palm",
    "peace_sign",
    "pointing",
    "thumbs_down",
    "thumbs_up",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DualStreamModel stub
# ---------------------------------------------------------------------------
class DualStreamModel:
    """
    Stub interface for the Architecture A dual-stream model.

    Replace this class with the real implementation once
    semantic_hgr.models.dual_stream is complete.

    Expected signature:
        model = DualStreamModel.from_checkpoint(path, device)
        predictions: List[Tuple[str, float]] = model.predict(image_bgr, top_k=5)
        # Each tuple is (class_name, confidence_score)
    """

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu"):
        if checkpoint_path is not None and checkpoint_path.exists():
            raise NotImplementedError(
                "DualStreamModel.from_checkpoint: real implementation pending.\n"
                "Run 'make train-dual' to produce a checkpoint, then update\n"
                "src/semantic_hgr/models/dual_stream.py."
            )
        # Allow checkpoint=None for framework-only testing
        instance = cls()
        instance._stub = True
        return instance

    def predict(self, image_bgr: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return uniform stub predictions (framework testing only)."""
        if not getattr(self, "_stub", False):
            raise RuntimeError("DualStreamModel not properly initialised")
        n = len(GESTURE_CLASSES)
        uniform = 1.0 / n
        ranked = sorted(
            [(cls, uniform) for cls in GESTURE_CLASSES],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Load static gesture dataset organised by class directory.

    Args:
        dataset_path: Root directory containing one subdirectory per class.

    Returns:
        Mapping from class name to list of image paths.
        Only directories matching GESTURE_CLASSES are included.
    """
    dataset: Dict[str, List[Path]] = defaultdict(list)

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in GESTURE_CLASSES:
            logger.warning("Skipping unrecognised class directory: %s", class_name)
            continue

        for img_path in class_dir.glob("*.jpg"):
            dataset[class_name].append(img_path)
        for img_path in class_dir.glob("*.png"):
            dataset[class_name].append(img_path)

    return dict(dataset)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_top_k_accuracy(
    predictions: List[List[Tuple[str, float]]],
    ground_truth: List[str],
    k: int,
) -> float:
    """Return Top-K accuracy as a fraction in [0, 1]."""
    if not ground_truth:
        return 0.0
    correct = sum(
        1
        for pred_list, true_label in zip(predictions, ground_truth)
        if true_label in [name for name, _ in pred_list[:k]]
    )
    return correct / len(ground_truth)


def compute_margin(predictions: List[Tuple[str, float]]) -> float:
    """Return confidence gap between top-1 and top-2 predictions."""
    if len(predictions) < 2:
        return 1.0
    return predictions[0][1] - predictions[1][1]


def build_confusion_matrix(
    predictions: List[List[Tuple[str, float]]],
    ground_truth: List[str],
    class_names: List[str],
) -> np.ndarray:
    """Build N×N confusion matrix from ranked prediction lists."""
    n = len(class_names)
    confusion = np.zeros((n, n), dtype=int)
    idx = {name: i for i, name in enumerate(class_names)}

    for pred_list, true_label in zip(predictions, ground_truth):
        if not pred_list:
            continue
        pred_label = pred_list[0][0]
        if true_label in idx and pred_label in idx:
            confusion[idx[true_label], idx[pred_label]] += 1

    return confusion


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Architecture A dual-stream model  (Top-K, confusion)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset root (class sub-directories)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to dual-stream checkpoint (.pt). Omit to test framework only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/static_eval_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1,3,5",
        help="Comma-separated K values",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device",
    )

    args = parser.parse_args()
    k_values = [int(k.strip()) for k in args.top_k.split(",")]

    logger.info("=" * 70)
    logger.info("ARCHITECTURE A — STATIC GESTURE EVALUATION")
    logger.info("=" * 70)
    logger.info("Dataset   : %s", args.dataset)
    logger.info("Checkpoint: %s", args.checkpoint or "None (stub mode)")
    logger.info("Top-K     : %s", k_values)
    logger.info("Device    : %s", args.device)

    # Load dataset
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No valid class directories found in %s", args.dataset)
        logger.error("Expected classes: %s", GESTURE_CLASSES)
        return

    total_images = sum(len(v) for v in dataset.values())
    logger.info("Classes loaded: %d  |  Total images: %d", len(dataset), total_images)
    for cls, imgs in dataset.items():
        logger.info("  %-18s %d images", cls, len(imgs))

    if total_images == 0:
        logger.error("No images found in dataset")
        return

    # Initialise model
    logger.info("Initialising DualStreamModel...")
    model = DualStreamModel.from_checkpoint(args.checkpoint, device=args.device)

    # Run inference
    logger.info("Running inference on %d images...", total_images)
    all_predictions: List[List[Tuple[str, float]]] = []
    all_ground_truth: List[str] = []
    all_margins: List[float] = []

    for class_name, image_paths in dataset.items():
        logger.info("  Processing: %s (%d images)", class_name, len(image_paths))
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Failed to load image: %s", img_path)
                continue
            try:
                preds = model.predict(img, top_k=max(k_values))
                all_predictions.append(preds)
                all_ground_truth.append(class_name)
                all_margins.append(compute_margin(preds))
            except Exception as exc:
                logger.error("Error processing %s: %s", img_path, exc)

    # Compute metrics
    logger.info("Computing metrics...")
    class_names_sorted = sorted(dataset.keys())
    results = {
        "model": "DualStreamModel (Architecture A — GCN + ViT-B/16)",
        "dataset_path": str(args.dataset),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "total_images": len(all_predictions),
        "num_classes": len(dataset),
        "class_names": GESTURE_CLASSES,
        "top_k_accuracy": {},
        "margin_stats": {},
        "per_class_accuracy": {},
        "confusion_matrix": {},
    }

    for k in k_values:
        acc = compute_top_k_accuracy(all_predictions, all_ground_truth, k)
        results["top_k_accuracy"][f"top_{k}"] = round(float(acc), 6)
        logger.info("Top-%d Accuracy: %.4f  (%.2f%%)", k, acc, acc * 100)

    if all_margins:
        margins = np.array(all_margins)
        results["margin_stats"] = {
            "mean": float(margins.mean()),
            "std": float(margins.std()),
            "min": float(margins.min()),
            "max": float(margins.max()),
            "median": float(np.median(margins)),
        }
        logger.info(
            "Confidence Margin — mean %.4f  std %.4f",
            margins.mean(),
            margins.std(),
        )

    for class_name in class_names_sorted:
        indices = [i for i, gt in enumerate(all_ground_truth) if gt == class_name]
        if indices:
            acc = compute_top_k_accuracy(
                [all_predictions[i] for i in indices],
                [all_ground_truth[i] for i in indices],
                k=1,
            )
            results["per_class_accuracy"][class_name] = round(float(acc), 6)

    confusion = build_confusion_matrix(
        all_predictions, all_ground_truth, class_names_sorted
    )
    results["confusion_matrix"] = {
        "class_order": class_names_sorted,
        "matrix": confusion.tolist(),
    }

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 70)
    logger.info("Evaluation complete.  Report saved: %s", args.output)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
