"""Build the LD-CONG Hybrid dataset by merging LD-CONG, Jester, and IPN NPZ files.

Merges:
  - LD-CONG NPZ (W=16): unique gestures + directional gestures (renamed to
    match Jester swipe labels).
  - Jester NPZ (W=32 → W=16 via center-crop): zoom classes + swipe augmentation.
  - IPN NPZ (W=32 → W=16 via center-crop): pointing and unknown background classes.

LD-CONG label remapping (from vocabulary_ld_cong_hybrid.yaml ``label_remap``):
  right    → swiping_right
  left     → swiping_left
  downward → swiping_down
  upward   → swiping_up

The ``shift_right`` class is dropped entirely (not present in the vocabulary).

Jester and IPN windows are W=32; they are truncated to W=16 by taking the
center-16 frames (frames 8:24) before merging with LD-CONG data.

Usage
-----
    python scripts/build_ld_cong_dataset.py \\
        --vocab-config   configs/gestures/vocabulary_ld_cong_hybrid.yaml \\
        --ld-cong-npz    data/processed/ld_cong_w16_dataset.npz \\
        --jester-npz     data/processed/jester_pretrain_dataset.npz \\
        --ipn-npz        data/processed/ipn_dataset_v5_10class_mirrored.npz \\
        --output-npz     data/processed/ld_cong_hybrid_dataset.npz

    # Rebuild even if output already exists:
    python scripts/build_ld_cong_dataset.py ... --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# NPZ loading helpers
# ---------------------------------------------------------------------------


def _load_source_npz(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays from a source NPZ into a plain dict."""
    logger.info("Loading %s …", path.name)
    with np.load(path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}
    if "labels" in data:
        data["labels"] = data["labels"].astype(str)
    if "recording_ids" in data:
        data["recording_ids"] = data["recording_ids"].astype(str)
    if "sample_ids" in data:
        data["sample_ids"] = data["sample_ids"].astype(str)
    n = len(data.get("labels", []))
    logger.info("  Loaded %d samples  keys=%s", n, sorted(data.keys()))
    return data


def _get_metadata(data: dict[str, np.ndarray]) -> dict:
    """Parse embedded metadata_json from a loaded NPZ dict."""
    if "metadata_json" in data:
        return json.loads(str(data["metadata_json"].item()))
    return {}


# ---------------------------------------------------------------------------
# Subset extraction
# ---------------------------------------------------------------------------


def _extract_class_subset(
    data: dict[str, np.ndarray],
    classes: list[str],
    *,
    source_prefix: str,
    max_samples_per_class: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return arrays restricted to samples whose label is in ``classes``."""
    labels = data["labels"]
    mask = np.isin(labels, classes)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        logger.warning(
            "No samples found for classes %s in source '%s'.", classes, source_prefix
        )
        return {}

    found = Counter(labels[indices].tolist())
    logger.info(
        "  %s — found classes: %s",
        source_prefix,
        {cls: found.get(cls, 0) for cls in classes},
    )

    if max_samples_per_class > 0:
        kept: list[int] = []
        for cls in classes:
            cls_idx = indices[labels[indices] == cls]
            if len(cls_idx) > max_samples_per_class:
                chosen = rng.choice(cls_idx, max_samples_per_class, replace=False)
                kept.extend(chosen.tolist())
                logger.info("    Capped %s: %d → %d", cls, len(cls_idx), max_samples_per_class)
            else:
                kept.extend(cls_idx.tolist())
        indices = np.array(sorted(kept), dtype=np.int64)

    n = len(indices)
    result: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        if key == "metadata_json":
            continue
        result[key] = arr[indices] if hasattr(arr, "__len__") and len(arr) == len(labels) else arr

    result["recording_ids"] = np.array(
        [f"{source_prefix}{rid}" for rid in result["recording_ids"]], dtype=object
    )
    result["sample_ids"] = np.array(
        [f"{source_prefix}{sid}" for sid in result["sample_ids"]], dtype=object
    )

    after = Counter(result["labels"].tolist())
    logger.info(
        "  %s — after cap: %s  (total %d)", source_prefix, dict(after), n
    )
    return result


# ---------------------------------------------------------------------------
# LD-CONG label remapping
# ---------------------------------------------------------------------------


def _remap_ld_cong_labels(
    data: dict[str, np.ndarray], remap: dict[str, str]
) -> dict[str, np.ndarray]:
    """Apply label remap in-place and return the modified dict.

    Labels not in ``remap`` are left unchanged.  After remapping, any samples
    whose label is not in the vocabulary will be dropped by
    ``_extract_class_subset`` naturally (they won't match any class).

    Note: labels are converted to object dtype before remapping to avoid
    numpy fixed-width unicode truncation (e.g. <U11 would silently truncate
    "swiping_right" to "swiping_righ").
    """
    # Use object dtype so new strings of any length are stored without truncation.
    labels = data["labels"].astype(object).copy()
    before = set(labels.tolist())
    for old, new in remap.items():
        labels[labels == old] = new
    after = set(labels.tolist())
    changed = before - after
    if changed:
        logger.info("  LD-CONG label remap applied: %s", remap)
        logger.info("  Labels removed from set: %s  |  Labels added: %s", changed, after - before)
    data = dict(data)  # shallow copy so we don't mutate the caller's dict
    data["labels"] = labels
    return data


# ---------------------------------------------------------------------------
# Jester window truncation (W=32 → W=target via center-crop)
# ---------------------------------------------------------------------------


def _center_crop_sequences(
    data: dict[str, np.ndarray], target_length: int
) -> dict[str, np.ndarray]:
    """Truncate sequences from the current window length to ``target_length``.

    Takes the center ``target_length`` frames.  Raises ``ValueError`` if the
    sequences are already shorter than ``target_length``.
    """
    seqs = data["sequences"]
    current_len = seqs.shape[1]
    if current_len == target_length:
        return data
    if current_len < target_length:
        raise ValueError(
            f"Cannot truncate sequences of length {current_len} to {target_length}: "
            "source is shorter than target."
        )
    start = (current_len - target_length) // 2
    end = start + target_length
    logger.info(
        "  Center-cropping sequences: W=%d → W=%d  (frames %d:%d)",
        current_len, target_length, start, end,
    )
    result = dict(data)
    result["sequences"] = seqs[:, start:end, :].copy()
    return result


# ---------------------------------------------------------------------------
# Merge and save
# ---------------------------------------------------------------------------


def _merge_subsets(
    subsets: list[dict[str, np.ndarray]],
    vocabulary: list[str],
    *,
    window_length: int,
    feature_dim: int,
    feature_schema_version: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Concatenate subset dicts and assign label_ids from the vocabulary."""
    n_total = sum(len(s["labels"]) for s in subsets if s)
    if n_total == 0:
        raise RuntimeError("No samples after merging — check vocabulary and source NPZs.")

    label_to_id = {lbl: idx for idx, lbl in enumerate(vocabulary)}

    all_arrays: dict[str, list[np.ndarray]] = {
        "sequences": [],
        "labels": [],
        "label_ids": [],
        "recording_ids": [],
        "sample_ids": [],
        "start_timestamp_ms": [],
        "end_timestamp_ms": [],
        "valid_ratio": [],
    }

    for subset in subsets:
        if not subset:
            continue
        all_arrays["sequences"].append(subset["sequences"])
        all_arrays["labels"].append(subset["labels"])
        ids = np.array(
            [label_to_id.get(lbl, -1) for lbl in subset["labels"].tolist()], dtype=np.int64
        )
        all_arrays["label_ids"].append(ids)
        all_arrays["recording_ids"].append(subset["recording_ids"])
        all_arrays["sample_ids"].append(subset["sample_ids"])
        n = len(subset["labels"])
        zeros_i64 = np.zeros(n, dtype=np.int64)
        ones_f32 = np.ones(n, dtype=np.float32)
        all_arrays["start_timestamp_ms"].append(subset.get("start_timestamp_ms", zeros_i64))
        all_arrays["end_timestamp_ms"].append(subset.get("end_timestamp_ms", zeros_i64))
        all_arrays["valid_ratio"].append(subset.get("valid_ratio", ones_f32))

    merged: dict[str, np.ndarray] = {
        key: np.concatenate(arrays, axis=0) for key, arrays in all_arrays.items() if arrays
    }

    bad = (merged["label_ids"] == -1).sum()
    if bad > 0:
        bad_labels = set(merged["labels"][merged["label_ids"] == -1].tolist())
        logger.warning("  %d samples have labels not in vocabulary: %s", bad, bad_labels)
        keep = merged["label_ids"] != -1
        merged = {k: v[keep] for k, v in merged.items()}
        logger.warning("  Removed %d invalid-label samples.", bad)

    seqs = merged["sequences"]
    if seqs.ndim != 3 or seqs.shape[1] != window_length or seqs.shape[2] != feature_dim:
        raise RuntimeError(
            f"Merged sequences shape {seqs.shape!r} does not match expected "
            f"(N, {window_length}, {feature_dim})."
        )

    metadata = {
        "feature_schema_version": feature_schema_version,
        "feature_dim": feature_dim,
        "window_length": window_length,
        "window_step": None,
        "min_valid_ratio": None,
        "label_strategy": "ld_cong_hybrid_merge",
        "vocabulary": vocabulary,
        "seed": seed,
    }
    merged["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    return merged


def _save_npz(data: dict[str, np.ndarray], output_path: Path) -> None:
    """Save merged arrays to a compressed NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    size_mb = output_path.stat().st_size / (1024**2)
    n = len(data["labels"])
    logger.info("Saved %d samples → %s  (%.1f MB)", n, output_path, size_mb)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge LD-CONG, Jester, and IPN NPZ datasets into a hybrid dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vocab-config",
        type=Path,
        default=Path("configs/gestures/vocabulary_ld_cong_hybrid.yaml"),
        help="Hybrid vocabulary YAML (defines classes, source provenance, and label remaps).",
    )
    p.add_argument(
        "--ld-cong-npz",
        type=Path,
        default=Path("data/processed/ld_cong_w16_dataset.npz"),
        help="Pre-built LD-CONG NPZ (W=16, output of the LD-CONG dataset prep notebook).",
    )
    p.add_argument(
        "--jester-npz",
        type=Path,
        default=Path("data/processed/jester_pretrain_dataset.npz"),
        help=(
            "Pre-built Jester NPZ (W=32).  Only the Jester classes in the vocabulary "
            "are extracted; windows are center-cropped to match LD-CONG's W=16."
        ),
    )
    p.add_argument(
        "--ipn-npz",
        type=Path,
        default=Path("data/processed/ipn_dataset_v5_10class_mirrored.npz"),
        help=(
            "Pre-built IPN NPZ (W=32).  Only the IPN classes in the vocabulary "
            "are extracted; windows are center-cropped to match LD-CONG's W=16."
        ),
    )
    p.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/processed/ld_cong_hybrid_dataset.npz"),
        help="Output path for the merged hybrid NPZ.",
    )
    p.add_argument(
        "--max-samples-per-class",
        type=int,
        default=3000,
        help="Per-class sample cap before merging. 0 = no cap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible per-class subsampling.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output NPZ if it already exists.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Guard ─────────────────────────────────────────────────────────────────
    if args.output_npz.exists() and not args.force:
        logger.info("Output already exists: %s  (use --force to overwrite)", args.output_npz)
        sys.exit(0)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    if not args.vocab_config.exists():
        logger.error("Vocabulary config not found: %s", args.vocab_config)
        sys.exit(1)

    with args.vocab_config.open(encoding="utf-8") as vf:
        vcfg = yaml.safe_load(vf) or {}

    vocabulary: list[str] = [str(g) for g in (vcfg.get("gestures") or [])]
    if not vocabulary:
        logger.error("No gestures defined in vocabulary config.")
        sys.exit(1)

    sources_map: dict[str, str] = {
        cls: str(src) for cls, src in (vcfg.get("sources") or {}).items()
    }
    # LD-CONG label remap (rename directional gestures to Jester swipe names).
    ld_cong_remap: dict[str, str] = vcfg.get("label_remap", {}).get("ld_cong", {})

    ld_cong_only_classes = [
        cls for cls in vocabulary if sources_map.get(cls, "ld_cong") == "ld_cong"
    ]
    jester_only_classes = [
        cls for cls in vocabulary if sources_map.get(cls, "ld_cong") == "jester"
    ]
    ipn_only_classes = [
        cls for cls in vocabulary if sources_map.get(cls, "ld_cong") == "ipn"
    ]
    both_classes = [
        cls for cls in vocabulary if sources_map.get(cls, "ld_cong") == "both"
    ]

    logger.info("Hybrid vocabulary: %d classes", len(vocabulary))
    logger.info("  LD-CONG only : %s", ld_cong_only_classes)
    logger.info("  Jester only  : %s", jester_only_classes)
    logger.info("  IPN only     : %s", ipn_only_classes)
    logger.info("  Both sources : %s", both_classes)
    logger.info("  LD-CONG label remap: %s", ld_cong_remap)

    rng = np.random.default_rng(args.seed)

    # ── LD-CONG data ──────────────────────────────────────────────────────────
    if not args.ld_cong_npz.exists():
        logger.error(
            "LD-CONG NPZ not found: %s\n"
            "  Build it first by running the LD-CONG dataset preparation notebook.",
            args.ld_cong_npz,
        )
        sys.exit(1)

    ld_cong_data = _load_source_npz(args.ld_cong_npz)
    ld_cong_meta = _get_metadata(ld_cong_data)
    window_length: int = int(ld_cong_data["sequences"].shape[1])
    feature_dim: int = int(ld_cong_data["sequences"].shape[2])
    feature_schema = ld_cong_meta.get("feature_schema_version", "feature-schema-v5")

    logger.info(
        "LD-CONG sequences: W=%d, F=%d, schema=%s", window_length, feature_dim, feature_schema
    )

    # Apply label remap before extraction.
    if ld_cong_remap:
        ld_cong_data = _remap_ld_cong_labels(ld_cong_data, ld_cong_remap)

    # Classes to extract from LD-CONG (unique + both).
    ld_cong_target_classes = ld_cong_only_classes + both_classes
    ld_cong_subset = _extract_class_subset(
        ld_cong_data,
        ld_cong_target_classes,
        source_prefix="ld_cong_",
        max_samples_per_class=args.max_samples_per_class,
        rng=rng,
    )
    del ld_cong_data

    # ── Jester data ───────────────────────────────────────────────────────────
    jester_classes_needed = jester_only_classes + both_classes
    jester_subset: dict[str, np.ndarray] = {}

    if jester_classes_needed:
        if not args.jester_npz.exists():
            logger.error(
                "Jester NPZ not found: %s\n"
                "  Build it first with: python scripts/build_jester_dataset.py",
                args.jester_npz,
            )
            sys.exit(1)

        jester_data = _load_source_npz(args.jester_npz)

        # Validate feature dimension compatibility.
        jester_feature_dim = int(jester_data["sequences"].shape[2])
        if jester_feature_dim != feature_dim:
            logger.error(
                "Feature dimension mismatch: LD-CONG F=%d vs Jester F=%d. "
                "Both must be built with the same feature schema.",
                feature_dim, jester_feature_dim,
            )
            sys.exit(1)

        # Center-crop Jester sequences from W=32 to match LD-CONG W=16.
        jester_data = _center_crop_sequences(jester_data, window_length)

        jester_subset = _extract_class_subset(
            jester_data,
            jester_classes_needed,
            source_prefix="jester_",
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )
        del jester_data
    else:
        logger.info("No Jester classes required — skipping Jester NPZ load.")

    # ── IPN data ──────────────────────────────────────────────────────────────
    ipn_subset: dict[str, np.ndarray] = {}

    if ipn_only_classes:
        if not args.ipn_npz.exists():
            logger.error(
                "IPN NPZ not found: %s\n"
                "  Build it first with: python scripts/build_ipn_dataset.py",
                args.ipn_npz,
            )
            sys.exit(1)

        ipn_data = _load_source_npz(args.ipn_npz)

        # Validate feature dimension compatibility.
        ipn_feature_dim = int(ipn_data["sequences"].shape[2])
        if ipn_feature_dim != feature_dim:
            logger.error(
                "Feature dimension mismatch: LD-CONG F=%d vs IPN F=%d. "
                "Both must be built with the same feature schema.",
                feature_dim, ipn_feature_dim,
            )
            sys.exit(1)

        # Center-crop IPN sequences from W=32 to match LD-CONG W=16.
        ipn_data = _center_crop_sequences(ipn_data, window_length)

        ipn_subset = _extract_class_subset(
            ipn_data,
            ipn_only_classes,
            source_prefix="ipn_",
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )
        del ipn_data
    else:
        logger.info("No IPN classes required — skipping IPN NPZ load.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = _merge_subsets(
        [ld_cong_subset, jester_subset, ipn_subset],
        vocabulary=vocabulary,
        window_length=window_length,
        feature_dim=feature_dim,
        feature_schema_version=feature_schema,
        seed=args.seed,
    )

    n_total = len(merged["labels"])
    dist = Counter(merged["labels"].tolist())
    logger.info("Merged dataset: %d total samples  %d classes", n_total, len(dist))
    for cls in vocabulary:
        cnt = dist.get(cls, 0)
        bar = "█" * min(40, cnt // max(1, n_total // 1200))
        logger.info("  %-35s %6d  %s", cls, cnt, bar)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_npz(merged, args.output_npz)

    logger.info("Done.")


if __name__ == "__main__":
    main()
