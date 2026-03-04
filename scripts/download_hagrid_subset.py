#!/usr/bin/env python3
"""
HaGRID Subset Download for Static Gesture Evaluation
=====================================================

Downloads 8 gesture classes from the HaGRID dataset via Hugging Face and
writes JPEG images into the directory structure expected by `make static-eval`:

    data/hagrid_subset/
        thumbs_up/       ← hand gesture JPEGs
        thumbs_down/
        open_palm/
        closed_fist/
        peace_sign/
        ok_sign/
        pointing/
        call_gesture/

Source dataset : cj-mills/hagrid-classification-512p-no-gesture-150k
  - 153,735 pre-cropped 512px images, already organised for classification
  - Schema: { "image": PIL.Image, "label": ClassLabel(int) }
  - Single "train" split; no bboxes needed (images already cropped)

Folder names use the system's intent vocabulary (not HaGRID class names).
`no_gesture` is intentionally excluded: MediaPipe gates that case upstream;
including it would silently deflate Top-K accuracy in static_eval.

HaGRID class → system intent name:
  call     → call_gesture      (thumb + pinky extended)
  dislike  → thumbs_down
  fist     → closed_fist
  like     → thumbs_up
  ok       → ok_sign
  one      → pointing          (index finger extended; v1 dataset uses "one")
  palm     → open_palm
  peace    → peace_sign

Requirements:
    pip install datasets          # Hugging Face streaming download
    pip install opencv-python     # Resize + JPEG save

Usage:
    python scripts/download_hagrid_subset.py
    python scripts/download_hagrid_subset.py --n-samples 500
    python scripts/download_hagrid_subset.py --output /custom/path --n-samples 1000
    python scripts/download_hagrid_subset.py --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Disable HuggingFace multiprocessing/shared-memory to avoid torch_shm_manager
# permission errors on restrictive macOS environments.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_VERBOSITY", "error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# HuggingFace dataset ID (single flat dataset, filter by label integer)
HF_DATASET_ID = "cj-mills/hagrid-classification-512p-no-gesture-150k"
HF_SPLIT = "train"

# HaGRID class name → system intent name (8 active classes).
# `one` = "one finger up" = index-finger-extended = pointing (v1 label name).
# `no_gesture`, `mute`, `four`, `stop`, `rock`, etc. are deliberately absent.
HAGRID_TO_INTENT: dict[str, str] = {
    "call":    "call_gesture",
    "dislike": "thumbs_down",
    "fist":    "closed_fist",
    "like":    "thumbs_up",
    "ok":      "ok_sign",
    "one":     "pointing",
    "palm":    "open_palm",
    "peace":   "peace_sign",
}

# JPEG quality for saved images.
JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_image) -> np.ndarray:
    """Convert PIL Image → BGR uint8 ndarray (OpenCV convention)."""
    rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def download_subset(
    output_dir: Path,
    n_samples: int,
    image_size: int,
    dry_run: bool,
) -> None:
    """
    Stream HaGRID from HuggingFace, filter to the 8 target classes, and
    save up to `n_samples` JPEG images per class under `output_dir`.
    """
    logger.info("=" * 68)
    logger.info("HaGRID Subset Download")
    logger.info("  Source  : %s", HF_DATASET_ID)
    logger.info("  Output  : %s", output_dir)
    logger.info("  Samples : %d per class  (%d total)", n_samples, n_samples * len(HAGRID_TO_INTENT))
    logger.info("  Size    : %d×%d px", image_size, image_size)
    logger.info("  Classes : %d  (no_hand excluded — handled by MediaPipe)", len(HAGRID_TO_INTENT))
    if dry_run:
        logger.info("  Mode    : DRY RUN — no files will be written")
    logger.info("=" * 68)

    if dry_run:
        for intent_name in HAGRID_TO_INTENT.values():
            logger.info("  [dry-run] would save %d images → %s/%s", n_samples, output_dir, intent_name)
        logger.info("=" * 68)
        logger.info("Dry run complete. No files written.")
        logger.info("=" * 68)
        return

    # ---- Import here so missing package gives a clean error ----
    try:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("'datasets' package not found.  Install with:  pip install datasets")
        sys.exit(1)

    # ---- Determine how many images each class already has ----
    counters: dict[str, int] = {}
    for intent_name in HAGRID_TO_INTENT.values():
        class_dir = output_dir / intent_name
        existing = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
        counters[intent_name] = existing
        if existing >= n_samples:
            logger.info("  %-15s already complete (%d / %d)", intent_name, existing, n_samples)

    remaining = {
        hagrid_cls: intent
        for hagrid_cls, intent in HAGRID_TO_INTENT.items()
        if counters[intent] < n_samples
    }
    if not remaining:
        logger.info("All classes already complete. Nothing to download.")
        logger.info("Run evaluation with:  make static-eval")
        logger.info("=" * 68)
        return

    logger.info("Classes still needed: %s", list(remaining.values()))

    # ---- Create output directories ----
    for intent_name in remaining.values():
        (output_dir / intent_name).mkdir(parents=True, exist_ok=True)

    # ---- Build label-name → intent mapping using the ClassLabel feature ----
    # Load a tiny slice first to get the label names without streaming the whole dataset
    logger.info("Loading dataset info …")
    try:
        ds_info = load_dataset(HF_DATASET_ID, split=HF_SPLIT, streaming=True)
        # Shuffle so all 19 classes are interleaved from the start.
        # Avoids scanning 70k+ samples before reaching alphabetically-later classes
        # like 'palm' (label 9) or 'peace' (label 10).
        ds_info = ds_info.shuffle(seed=42, buffer_size=8000)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    # Resolve ClassLabel names → index mapping
    features = ds_info.features
    label_feature = features.get("label")
    if label_feature is None:
        logger.error("'label' field not found in dataset features. Schema: %s", list(features.keys()))
        sys.exit(1)

    # ClassLabel stores names as label_feature.names (list indexed by int value)
    label_names: list[str] = label_feature.names  # e.g. ["call", "dislike", ...]
    logger.info("Dataset label names (%d total): %s", len(label_names), label_names)

    # Map HaGRID class name → integer label index
    hagrid_cls_to_idx: dict[str, int] = {
        name: idx for idx, name in enumerate(label_names)
    }
    target_indices: dict[int, str] = {}  # label_int → intent_name (for quick lookup)
    for hagrid_cls, intent_name in remaining.items():
        idx = hagrid_cls_to_idx.get(hagrid_cls)
        if idx is None:
            logger.warning("  HaGRID class '%s' not found in dataset labels — skipping.", hagrid_cls)
            continue
        target_indices[idx] = intent_name
        logger.info("  Label %2d  '%-15s' → '%s'", idx, hagrid_cls, intent_name)

    if not target_indices:
        logger.error("None of the target HaGRID classes found in dataset. Aborting.")
        sys.exit(1)

    # ---- Stream through and collect ----
    total_needed = sum(
        n_samples - counters[intent] for intent in remaining.values()
    )
    logger.info("Streaming dataset (need %d more images total) …", total_needed)

    processed = 0
    total_saved = sum(counters.values())

    for sample in ds_info:
        # Check if all target classes are fulfilled
        if all(counters[intent] >= n_samples for intent in remaining.values()):
            break

        label_int = sample.get("label")
        if label_int not in target_indices:
            continue  # Not a class we need

        intent_name = target_indices[label_int]
        if counters[intent_name] >= n_samples:
            continue  # This class is already done

        pil_image = sample.get("image")
        if pil_image is None:
            continue

        bgr = _pil_to_bgr(pil_image)

        if image_size != bgr.shape[0] or image_size != bgr.shape[1]:
            bgr = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        idx_in_class = counters[intent_name]
        out_path = output_dir / intent_name / f"{intent_name}_{idx_in_class:05d}.jpg"
        ok = cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            logger.warning("  cv2.imwrite failed for %s", out_path)
            continue

        counters[intent_name] += 1
        total_saved += 1
        processed += 1

        # Progress every 100 successful saves
        if processed % 100 == 0:
            status = "  ".join(
                f"{intent}={counters[intent]}/{n_samples}"
                for intent in remaining.values()
            )
            logger.info("  Progress: %s", status)

    # ---- Final summary ----
    logger.info("=" * 68)
    total_expected = n_samples * len(HAGRID_TO_INTENT)
    final_total = sum(
        len(list((output_dir / intent).glob("*.jpg")))
        for intent in HAGRID_TO_INTENT.values()
        if (output_dir / intent).exists()
    )
    for hagrid_cls, intent_name in HAGRID_TO_INTENT.items():
        count = len(list((output_dir / intent_name).glob("*.jpg"))) if (output_dir / intent_name).exists() else 0
        status = "✓" if count >= n_samples else f"INCOMPLETE ({count}/{n_samples})"
        logger.info("  %-15s %s", intent_name, status)
    logger.info("Download complete: %d / %d images saved.", final_total, total_expected)
    logger.info("Dataset ready at: %s", output_dir)
    logger.info("Run evaluation with:  make static-eval")
    logger.info("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HaGRID subset for static gesture evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hagrid_subset"),
        help="Root directory for the downloaded subset.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=750,
        help="Number of images per class (500 = minimum reliable, 1000 = stable).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=(
            "Side length (px) for saved JPEGs. 224 matches ViT-B/16 input size "
            "used by the Architecture A RGB stream (timm vit_base_patch16_224)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    download_subset(
        output_dir=args.output,
        n_samples=args.n_samples,
        image_size=args.image_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
