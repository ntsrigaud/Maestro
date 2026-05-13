"""Merge pre-built Jester and IPN NPZ datasets into a hybrid 10-class dataset.

The hybrid dataset combines:
  - 7 classes from Jester (swiping_*, stop_sign, zooming_*_full_hand)
  - 3 classes from IPN    (unknown/D0X background, point_one, point_two)

This script operates on **pre-built NPZ files** produced by
``build_jester_dataset.py`` and ``build_ipn_dataset.py`` respectively.  It
does NOT re-run MediaPipe landmark extraction.  Both source NPZs must exist
before this script is called.

Source provenance is read from the ``sources`` section of the vocabulary YAML::

    sources:
      unknown:   ipn
      point_one: ipn
      ...

Any class listed as ``ipn`` is taken from the IPN NPZ.  Classes absent from
``sources`` (or listed as ``jester``) default to the Jester source.  Jester
data can be provided either as a single pre-built NPZ (``--jester-npz``) or as
a directory of per-class chunk NPZs (``--jester-chunk-dir``) produced by
``build_jester_dataset.py --chunk-dir``.  The chunk-dir approach is strongly
preferred because it loads only the 7 needed Jester class files instead of
the full ~27-class Jester NPZ, avoiding OOM kills on memory-constrained hosts.

Per-class balancing
-------------------
A ``--max-samples-per-class`` cap is applied **after** merging.  Jester
classes can have 3,000–8,000 windows in the subset while IPN pointing classes
can exceed 10,000.  The cap (default: 5,000) keeps training balanced.
Samples are drawn randomly using ``--seed`` for reproducibility.

Recording-ID namespacing
-------------------------
IPN recording IDs are prefixed with ``ipn_`` and Jester IDs with ``jester_``
to prevent collision.  This ensures the downstream ``GestureDataModule``
stratified split keeps all windows from the same source video in the same
split.

Output format
-------------
The output NPZ is fully compatible with ``GestureDataModule`` and the existing
training pipeline.  Keys:
  sequences        (N, T, F) float32
  labels           (N,)      str
  label_ids        (N,)      int64
  recording_ids    (N,)      str
  sample_ids       (N,)      str
  start_timestamp_ms (N,)    int64
  end_timestamp_ms   (N,)    int64
  valid_ratio      (N,)      float32
  metadata_json              JSON string embedding the hybrid vocabulary

Usage
-----
    # Preferred: use per-class Jester chunks (low memory):
    python scripts/build_hybrid_dataset.py \\
        --vocab-config   configs/gestures/vocabulary_hybrid.yaml \\
        --jester-chunk-dir data/processed/jester_chunks \\
        --ipn-npz        data/processed/ipn_dataset_v5_10class_mirrored.npz \\
        --output-npz     data/processed/hybrid_dataset.npz

    # Alternative: use a single pre-built Jester NPZ:
    python scripts/build_hybrid_dataset.py \\
        --vocab-config configs/gestures/vocabulary_hybrid.yaml \\
        --jester-npz   data/processed/jester_pretrain_dataset.npz \\
        --ipn-npz      data/processed/ipn_dataset_v5_10class_mirrored.npz \\
        --output-npz   data/processed/hybrid_dataset.npz

    # Rebuild even if output already exists:
    python scripts/build_hybrid_dataset.py ... --force
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
    # Ensure labels are Python str objects, not numpy object arrays of bytes.
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
# Per-class Jester chunk loader
# ---------------------------------------------------------------------------


def _load_jester_from_chunks(
    chunk_dir: Path,
    classes: list[str],
    *,
    max_samples_per_class: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Load only the requested Jester classes from a per-class chunk directory.

    Each chunk file is expected at ``chunk_dir/{class_name}.npz`` and was
    produced by ``build_jester_dataset.py --chunk-dir``.  Only the files for
    the *classes* needed by the hybrid vocabulary are loaded, keeping memory
    usage proportional to the number of hybrid Jester classes (7) rather than
    the full 27-class Jester vocabulary.

    Recording and sample IDs are prefixed with ``jester_`` to match the
    namespacing convention used by ``_extract_class_subset``.
    """
    all_seqs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_rids: list[np.ndarray] = []
    all_sids: list[np.ndarray] = []
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []

    for cls in classes:
        chunk_path = chunk_dir / f"{cls}.npz"
        if not chunk_path.exists():
            logger.warning(
                "Jester chunk not found for class %r: %s — skipped.", cls, chunk_path
            )
            continue
        with np.load(chunk_path, allow_pickle=True) as raw:
            seqs = raw["sequences"].astype(np.float32)
            lbs = raw["labels"].astype(str)
            rids = raw["recording_ids"].astype(str)
            sids = raw["sample_ids"].astype(str)
            n = len(lbs)
            zeros = np.zeros(n, dtype=np.int64)
            ones = np.ones(n, dtype=np.float32)
            starts = raw["start_timestamp_ms"].astype(np.int64) if "start_timestamp_ms" in raw else zeros
            ends = raw["end_timestamp_ms"].astype(np.int64) if "end_timestamp_ms" in raw else zeros
            valid = raw["valid_ratio"].astype(np.float32) if "valid_ratio" in raw else ones

        # Per-class cap.
        if max_samples_per_class > 0 and n > max_samples_per_class:
            idx = rng.choice(n, max_samples_per_class, replace=False)
            seqs, lbs, rids, sids = seqs[idx], lbs[idx], rids[idx], sids[idx]
            starts, ends, valid = starts[idx], ends[idx], valid[idx]
            n = max_samples_per_class
            logger.info("    Capped %s (chunk): %d → %d", cls, len(lbs) + (n - max_samples_per_class if max_samples_per_class else 0), n)

        logger.info("  jester_chunk: %s  %d samples", cls, n)
        all_seqs.append(seqs)
        all_labels.append(lbs)
        all_rids.append(np.array([f"jester_{rid}" for rid in rids], dtype=object))
        all_sids.append(np.array([f"jester_{sid}" for sid in sids], dtype=object))
        all_start.append(starts)
        all_end.append(ends)
        all_valid.append(valid)

    if not all_seqs:
        logger.warning("No Jester chunk data loaded for classes: %s", classes)
        return {}

    result: dict[str, np.ndarray] = {
        "sequences": np.concatenate(all_seqs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "recording_ids": np.concatenate(all_rids, axis=0),
        "sample_ids": np.concatenate(all_sids, axis=0),
        "start_timestamp_ms": np.concatenate(all_start, axis=0),
        "end_timestamp_ms": np.concatenate(all_end, axis=0),
        "valid_ratio": np.concatenate(all_valid, axis=0),
    }
    # label_ids will be recomputed in _merge_subsets from the hybrid vocabulary.
    result["label_ids"] = np.zeros(len(result["labels"]), dtype=np.int64)
    return result


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
    """Return arrays restricted to samples whose label is in ``classes``.

    Parameters
    ----------
    source_prefix:
        Prepended to every ``recording_id`` and ``sample_id`` to prevent
        collisions when multiple source datasets are merged.
    max_samples_per_class:
        Per-class sample cap applied *before* merging.  Set to 0 to disable.
    """
    labels = data["labels"]  # (N,) str
    mask = np.isin(labels, classes)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        logger.warning(
            "No samples found for classes %s in source with prefix '%s'.",
            classes,
            source_prefix,
        )
        return {}

    found = Counter(labels[indices].tolist())
    logger.info(
        "  %s — found classes: %s",
        source_prefix,
        {cls: found.get(cls, 0) for cls in classes},
    )

    # Per-class cap — subsample each class independently.
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

    result: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        if key == "metadata_json":
            continue
        result[key] = arr[indices] if hasattr(arr, "__len__") and len(arr) == len(labels) else arr

    # Prefix recording and sample IDs.
    n = len(indices)
    result["recording_ids"] = np.array(
        [f"{source_prefix}{rid}" for rid in result["recording_ids"]], dtype=object
    )
    result["sample_ids"] = np.array(
        [f"{source_prefix}{sid}" for sid in result["sample_ids"]], dtype=object
    )

    after = Counter(result["labels"].tolist())
    logger.info(
        "  %s — after cap: %s  (total %d)",
        source_prefix,
        dict(after),
        n,
    )
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
    """Concatenate subset dicts and assign label_ids from hybrid vocabulary."""
    n_total = sum(len(s["labels"]) for s in subsets if s)
    if n_total == 0:
        raise RuntimeError("No samples after merging — check vocabulary and source NPZs.")

    label_to_id = {lbl: idx for idx, lbl in enumerate(vocabulary)}

    # Build a merge order (shuffle at recording level handled by GestureDataModule).
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
        # Recompute label_ids from hybrid vocab (source IDs differ).
        ids = np.array(
            [label_to_id.get(lbl, -1) for lbl in subset["labels"].tolist()], dtype=np.int64
        )
        all_arrays["label_ids"].append(ids)
        all_arrays["recording_ids"].append(subset["recording_ids"])
        all_arrays["sample_ids"].append(subset["sample_ids"])
        # Fallback for optional fields.
        n = len(subset["labels"])
        zeros_i64 = np.zeros(n, dtype=np.int64)
        ones_f32 = np.ones(n, dtype=np.float32)
        all_arrays["start_timestamp_ms"].append(subset.get("start_timestamp_ms", zeros_i64))
        all_arrays["end_timestamp_ms"].append(subset.get("end_timestamp_ms", zeros_i64))
        all_arrays["valid_ratio"].append(subset.get("valid_ratio", ones_f32))

    merged: dict[str, np.ndarray] = {
        key: np.concatenate(arrays, axis=0) for key, arrays in all_arrays.items() if arrays
    }

    # Warn about any unknown labels.
    bad = (merged["label_ids"] == -1).sum()
    if bad > 0:
        bad_labels = set(merged["labels"][merged["label_ids"] == -1].tolist())
        logger.warning("  %d samples have labels not in vocabulary: %s", bad, bad_labels)
        # Remove them.
        keep = merged["label_ids"] != -1
        merged = {k: v[keep] for k, v in merged.items()}
        logger.warning("  Removed %d invalid-label samples.", bad)

    # Validate shape.
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
        "window_step": None,  # mixed sources
        "min_valid_ratio": None,
        "label_strategy": "hybrid_merge",
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
        description=("Merge Jester and IPN NPZ datasets into a hybrid 10-class dataset."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vocab-config",
        type=Path,
        default=Path("configs/gestures/vocabulary_hybrid.yaml"),
        help="Hybrid vocabulary YAML (defines classes + source provenance).",
    )
    p.add_argument(
        "--jester-npz",
        type=Path,
        default=None,
        help=(
            "Pre-built full Jester NPZ (output of build_jester_dataset.py).  "
            "Provide either this or --jester-chunk-dir (not both)."
        ),
    )
    p.add_argument(
        "--jester-chunk-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory of per-class Jester chunk NPZs produced by "
            "'build_jester_dataset.py --chunk-dir'.  Only the files for the "
            "Jester classes in the hybrid vocabulary are loaded, which is much "
            "more memory-efficient than loading the full Jester NPZ."
        ),
    )
    p.add_argument(
        "--ipn-npz",
        type=Path,
        default=Path("data/processed/ipn_dataset_v5_10class.npz"),
        help="Pre-built IPN NPZ (output of build_ipn_dataset.py with 10-class vocab).",
    )
    p.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/processed/hybrid_dataset.npz"),
        help="Output path for the merged hybrid NPZ.",
    )
    p.add_argument(
        "--max-samples-per-class",
        type=int,
        default=5000,
        help=(
            "Per-class sample cap applied before merging. 0 = no cap (use all available samples)."
        ),
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

    # ── Guard ────────────────────────────────────────────────────────────────
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

    # Sources map: class_name → "jester" | "ipn" (defaults to "jester").
    sources_map: dict[str, str] = {
        cls: str(src) for cls, src in (vcfg.get("sources") or {}).items()
    }
    jester_classes = [cls for cls in vocabulary if sources_map.get(cls, "jester") == "jester"]
    ipn_classes = [cls for cls in vocabulary if sources_map.get(cls, "jester") == "ipn"]

    logger.info("Hybrid vocabulary: %d classes", len(vocabulary))
    logger.info("  Jester-sourced: %s", jester_classes)
    logger.info("  IPN-sourced   : %s", ipn_classes)

    rng = np.random.default_rng(args.seed)

    # ── Jester data — chunk dir or full NPZ ───────────────────────────────────
    use_chunks = args.jester_chunk_dir is not None
    if not use_chunks and args.jester_npz is None:
        logger.error(
            "Provide either --jester-chunk-dir or --jester-npz (neither was set)."
        )
        sys.exit(1)
    if use_chunks and args.jester_npz is not None:
        logger.warning(
            "Both --jester-chunk-dir and --jester-npz provided; "
            "--jester-chunk-dir takes precedence."
        )

    if use_chunks:
        if not args.jester_chunk_dir.is_dir():
            logger.error(
                "Jester chunk directory not found: %s\n"
                "  Build chunks first with: python scripts/build_jester_dataset.py --chunk-dir",
                args.jester_chunk_dir,
            )
            sys.exit(1)
        logger.info("Loading Jester classes from chunks: %s", args.jester_chunk_dir)
        jester_subset = _load_jester_from_chunks(
            chunk_dir=args.jester_chunk_dir,
            classes=jester_classes,
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )
        if not jester_subset:
            logger.error("No Jester data loaded from chunks — aborting.")
            sys.exit(1)
        seqs_shape = jester_subset["sequences"].shape
        feature_schema = "feature-schema-v5"
    else:
        if not args.jester_npz.exists():
            logger.error(
                "Jester NPZ not found: %s\n"
                "  Build it first with: python scripts/build_jester_dataset.py",
                args.jester_npz,
            )
            sys.exit(1)
        jester_data = _load_source_npz(args.jester_npz)
        seqs_shape = jester_data["sequences"].shape
        jester_meta = _get_metadata(jester_data)
        feature_schema = jester_meta.get("feature_schema_version", "feature-schema-v5")
        jester_subset = _extract_class_subset(
            jester_data,
            jester_classes,
            source_prefix="jester_",
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )
        del jester_data  # free memory after extraction

    window_length: int = int(seqs_shape[1])
    feature_dim: int = int(seqs_shape[2])

    # ── IPN data ──────────────────────────────────────────────────────────────
    if ipn_classes and not args.ipn_npz.exists():
        logger.error(
            "IPN NPZ not found: %s\n  Build it first with: python scripts/build_ipn_dataset.py",
            args.ipn_npz,
        )
        sys.exit(1)

    ipn_data = _load_source_npz(args.ipn_npz) if ipn_classes else {}

    # Validate IPN shape compatibility.
    if ipn_data:
        ipn_shape = ipn_data["sequences"].shape
        if ipn_shape[1] != window_length or ipn_shape[2] != feature_dim:
            logger.error(
                "Shape mismatch: Jester (N, %d, %d) vs IPN (N, %d, %d). "
                "Ensure both datasets were built with identical window_length "
                "and feature_schema.",
                window_length,
                feature_dim,
                ipn_shape[1],
                ipn_shape[2],
            )
            sys.exit(1)

    # ── Extract IPN subset ────────────────────────────────────────────────────
    ipn_subset = {}
    if ipn_classes and ipn_data:
        ipn_subset = _extract_class_subset(
            ipn_data,
            ipn_classes,
            source_prefix="ipn_",
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = _merge_subsets(
        [jester_subset, ipn_subset],
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
        logger.info("  %-30s %6d  %s", cls, cnt, bar)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_npz(merged, args.output_npz)

    # ── Metadata sidecar ──────────────────────────────────────────────────────
    meta_path = args.output_npz.with_suffix("").parent / (args.output_npz.stem + ".metadata.json")
    meta_path.write_text(
        json.dumps(
            {
                "vocabulary": vocabulary,
                "sources": sources_map,
                "jester_source": (
                    str(args.jester_chunk_dir) if use_chunks else str(args.jester_npz)
                ),
                "ipn_npz": str(args.ipn_npz),
                "max_samples_per_class": args.max_samples_per_class,
                "window_length": window_length,
                "feature_dim": feature_dim,
                "seed": args.seed,
                "total_samples": n_total,
                "class_distribution": dict(dist),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Metadata sidecar → %s", meta_path)


if __name__ == "__main__":
    main()

