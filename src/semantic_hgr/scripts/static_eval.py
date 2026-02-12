#!/usr/bin/env python3
"""
Static Gesture Evaluation - Top-K Retrieval Accuracy

Purpose: Evaluate static gesture recognition using retrieval-style metrics
         appropriate for zero-shot CLIP-based systems.

Metrics Computed:
- Top-1/3/5 Retrieval Accuracy
- Inter-class confidence margin
- Confusion matrix (for error analysis)

Usage:
    python scripts/static_eval.py \\
        --dataset path/to/hagrid_subset \\
        --output results/static_eval_report.json \\
        --top-k 1,3,5

Dataset Format:
    dataset/
        thumbs_up/
            image001.jpg
            image002.jpg
        thumbs_down/
            ...
        open_palm/
            ...
        (9 HaGRID-aligned static gesture classes)

Output:
    JSON report with:
    - Top-K accuracy scores
    - Per-class accuracy breakdown
    - Confusion matrix
    - Margin distribution statistics

Note: This evaluation is valid ONLY for static gestures. Dynamic gesture
      evaluation requires temporal modeling (ActionCLIP, VideoCLIP).
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import cv2

from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor, STATIC_SEMANTIC_INTENTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Load static gesture dataset organized by class.
    
    Args:
        dataset_path: Root directory with class subdirectories
    
    Returns:
        Dict mapping class names to list of image paths
    """
    dataset = defaultdict(list)
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in STATIC_SEMANTIC_INTENTS:
            logger.warning(f"Skipping unknown class: {class_name}")
            continue
        
        for img_path in class_dir.glob("*.jpg"):
            dataset[class_name].append(img_path)
        for img_path in class_dir.glob("*.png"):
            dataset[class_name].append(img_path)
    
    return dict(dataset)


def compute_top_k_accuracy(
    predictions: List[List[Tuple[str, float]]],
    ground_truth: List[str],
    k: int
) -> float:
    """
    Compute Top-K retrieval accuracy.
    
    Args:
        predictions: List of ranked intent lists (intent_name, confidence)
        ground_truth: List of true intent labels
        k: Number of top predictions to consider
    
    Returns:
        Accuracy as fraction in [0, 1]
    """
    correct = 0
    total = len(ground_truth)
    
    for pred_list, true_label in zip(predictions, ground_truth):
        top_k_intents = [intent for intent, _ in pred_list[:k]]
        if true_label in top_k_intents:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_margin(predictions: List[Tuple[str, float]]) -> float:
    """
    Compute confidence margin between top-1 and top-2 predictions.
    
    Args:
        predictions: Ranked list of (intent_name, confidence) tuples
    
    Returns:
        Margin = confidence(top-1) - confidence(top-2)
    """
    if len(predictions) < 2:
        return 1.0  # Only one prediction, maximum margin
    
    return predictions[0][1] - predictions[1][1]


def build_confusion_matrix(
    predictions: List[List[Tuple[str, float]]],
    ground_truth: List[str],
    class_names: List[str]
) -> np.ndarray:
    """
    Build confusion matrix from predictions.
    
    Args:
        predictions: List of ranked intent lists
        ground_truth: List of true labels
        class_names: Ordered list of all class names
    
    Returns:
        Confusion matrix (N x N) where N = len(class_names)
    """
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for pred_list, true_label in zip(predictions, ground_truth):
        if not pred_list:  # No prediction
            continue
        
        pred_label = pred_list[0][0]  # Top-1 prediction
        
        if true_label in class_to_idx and pred_label in class_to_idx:
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            confusion[true_idx, pred_idx] += 1
    
    return confusion


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate static gesture recognition (Top-K retrieval accuracy)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory (class subdirectories)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="results/static_eval_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1,3,5",
        help="Comma-separated list of K values for Top-K accuracy"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device"
    )
    
    args = parser.parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.top_k.split(",")]
    
    logger.info("=" * 70)
    logger.info("STATIC GESTURE EVALUATION - TOP-K RETRIEVAL ACCURACY")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Top-K values: {k_values}")
    logger.info(f"Device: {args.device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset)
    
    if not dataset:
        logger.error(f"No valid classes found in {args.dataset}")
        logger.error(f"Expected classes: {list(STATIC_SEMANTIC_INTENTS.keys())}")
        return
    
    logger.info(f"Loaded {len(dataset)} classes:")
    total_images = 0
    for class_name, images in dataset.items():
        logger.info(f"  {class_name}: {len(images)} images")
        total_images += len(images)
    
    if total_images == 0:
        logger.error("No images found in dataset")
        return
    
    # Initialize models
    logger.info("Initializing CLIP encoder and intent extractor...")
    encoder = CLIPEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device=args.device
    )
    
    extractor = SemanticIntentExtractor(
        clip_model=encoder.model,
        device=encoder.device,
        static_only=True  # CRITICAL: Static gestures only
    )
    
    # Run evaluation
    logger.info(f"Evaluating {total_images} images...")
    
    all_predictions = []
    all_ground_truth = []
    all_margins = []
    
    for class_name, image_paths in dataset.items():
        logger.info(f"Processing class: {class_name}...")
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load: {img_path}")
                continue
            
            # Encode image
            try:
                embedding = encoder.encode_image(img)
                
                # Extract intents
                predictions = extractor.extract_intent(
                    embedding,
                    top_k=max(k_values),
                    min_confidence=0.0
                )
                
                all_predictions.append(predictions)
                all_ground_truth.append(class_name)
                
                # Compute margin
                margin = compute_margin(predictions)
                all_margins.append(margin)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    results = {
        "dataset_path": str(args.dataset),
        "total_images": len(all_predictions),
        "num_classes": len(dataset),
        "class_names": list(dataset.keys()),
        "top_k_accuracy": {},
        "margin_stats": {},
        "per_class_accuracy": {},
        "confusion_matrix": None
    }
    
    # Top-K accuracy
    for k in k_values:
        acc = compute_top_k_accuracy(all_predictions, all_ground_truth, k)
        results["top_k_accuracy"][f"top_{k}"] = float(acc)
        logger.info(f"Top-{k} Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    
    # Margin statistics
    margins_array = np.array(all_margins)
    results["margin_stats"] = {
        "mean": float(margins_array.mean()),
        "std": float(margins_array.std()),
        "min": float(margins_array.min()),
        "max": float(margins_array.max()),
        "median": float(np.median(margins_array))
    }
    
    logger.info(f"Confidence Margin - Mean: {margins_array.mean():.4f}, "
                f"Std: {margins_array.std():.4f}")
    
    # Per-class accuracy
    for class_name in dataset.keys():
        class_indices = [i for i, gt in enumerate(all_ground_truth) if gt == class_name]
        class_predictions = [all_predictions[i] for i in class_indices]
        class_gt = [all_ground_truth[i] for i in class_indices]
        
        if class_predictions:
            top1_acc = compute_top_k_accuracy(class_predictions, class_gt, k=1)
            results["per_class_accuracy"][class_name] = float(top1_acc)
        else:
            results["per_class_accuracy"][class_name] = 0.0
    
    # Confusion matrix
    class_names_sorted = sorted(dataset.keys())
    confusion = build_confusion_matrix(
        all_predictions,
        all_ground_truth,
        class_names_sorted
    )
    results["confusion_matrix"] = {
        "class_order": class_names_sorted,
        "matrix": confusion.tolist()
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 70)
    logger.info(f"Evaluation complete. Report saved to: {args.output}")
    logger.info("=" * 70)
    
    # Print summary
    logger.info("SUMMARY:")
    for k in k_values:
        acc = results["top_k_accuracy"][f"top_{k}"]
        logger.info(f"  Top-{k} Accuracy: {acc * 100:.2f}%")
    logger.info(f"  Mean Margin: {results['margin_stats']['mean']:.4f}")
    logger.info(f"  Total Samples: {results['total_images']}")


if __name__ == "__main__":
    main()
