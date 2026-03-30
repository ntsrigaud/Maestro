"""Build a sliding-window LSTM training dataset from IPN Hand database videos.

The script implements a two-phase architecture:

  Phase 1 — Landmark Extraction (parallelised, with per-video caching)
    Each video is processed exactly once.  MediaPipe Hands runs on every frame
    in source order so the temporal tracker behaves correctly.  Results are
    saved to ``<ipn_root>/cache/<video_stem>.npz`` and reused on subsequent
    runs.

  Phase 2 — Dataset Assembly
    The annotation file is used to identify gesture segments.  IPN labels are
    mapped to the production vocabulary defined in ``vocabulary_ipn.yaml``.
    Segments whose IPN label has no mapping (G01, G02, G08, G09) are skipped.
    The ``unknown`` class (D0X, B0A, B0B) is subsampled to prevent it from
    dominating the dataset.  Frames for each segment are fed sequentially to
    ``SequenceDatasetBuilder``, which applies the existing feature transformer
    and sliding-window logic before saving the final NPZ.

Usage
-----
    python scripts/build_ipn_dataset.py \\
        --ipn-root data/IPN \\
        --output-npz data/processed/ipn_dataset.npz \\
        --num-workers 4

    # Force re-extraction (ignore caches):
    python scripts/build_ipn_dataset.py \\
        --ipn-root data/IPN \\
        --output-npz data/processed/ipn_dataset.npz \\
        --no-cache

    # Use original IPN train/test split instead of re-splitting:
    python scripts/build_ipn_dataset.py \\
        --ipn-root data/IPN \\
        --output-npz data/processed/ipn_train.npz \\
        --annotation-file Annot_TrainList.txt

Compatibility
-------------
The output NPZ is fully compatible with ``GestureDataModule`` and the
existing training pipeline in ``train_model.py``.  Pass the output path
to ``--dataset-path`` and set ``--vocab-config
configs/gestures/vocabulary_ipn.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a plain script from any cwd.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gesture_control.domain.entities import FramePacket  # noqa: E402
from gesture_control.infrastructure.mediapipe.hands_extractor import (  # noqa: E402
    MediaPipeHandsExtractor,
)
from gesture_control.training.dataset_builder import (  # noqa: E402
    LabeledLandmarkFrame,
    SequenceDatasetBuilder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPN label → production vocabulary mapping
# ---------------------------------------------------------------------------
# Gestures with no clear presentation-control semantic (click / double-click
# styles: G01, G02, G08, G09) are deliberately omitted — their frames will be
# skipped during assembly.  Pointing poses (B0A, B0B) are retained as
# negative examples under the ``unknown`` class.

IPN_TO_VOCAB: dict[str, str] = {
    "D0X": "unknown",  # non-gesture background
    "B0A": "unknown",  # pointing with one finger → negative example
    "B0B": "unknown",  # pointing with two fingers → negative example
    "G03": "swipe_up",  # throw up
    "G04": "swipe_down",  # throw down
    "G05": "swipe_left",  # throw left
    "G06": "swipe_right",  # throw right
    "G07": "open_palm_hold",  # open twice (closest to open-palm hold)
    "G10": "zoom_in",  # zoom in
    "G11": "zoom_out",  # zoom out
    # G01 click one finger  — skipped (no directional / hold semantic)
    # G02 click two fingers — skipped
    # G08 double click one  — skipped
    # G09 double click two  — skipped
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnnotationRow:
    video: str
    ipn_label: str
    label_id: int
    t_start: int  # 1-indexed, inclusive
    t_end: int  # 1-indexed, inclusive
    n_frames: int


@dataclass
class VideoLandmarks:
    """Per-frame MediaPipe results for a single video."""

    video_stem: str
    fps: float
    frame_ids: np.ndarray  # shape (N,) int32, 1-indexed
    valid: np.ndarray  # shape (N,) bool
    landmarks: np.ndarray  # shape (N, 21, 3) float32
    detect_conf: np.ndarray  # shape (N,) float32
    track_conf: np.ndarray  # shape (N,) float32
    handedness: list[str | None]  # length N


# ---------------------------------------------------------------------------
# Phase 1 — per-video landmark extraction (worker)
# ---------------------------------------------------------------------------


def _extract_video_worker(args: tuple[str, str, float, float]) -> tuple[str, str]:
    """Worker function: extract MediaPipe landmarks for a single video.

    Parameters
    ----------
    args:
        Tuple of (video_path, cache_path, min_detection_confidence,
        min_tracking_confidence).

    Returns
    -------
    (video_stem, cache_path)
        The stem of the processed video and the path where results were saved.
        On failure the stem is returned with a cache_path of empty string so
        the caller can handle it.
    """
    video_path_str, cache_path_str, min_det_conf, min_track_conf = args
    video_path = Path(video_path_str)
    cache_path = Path(cache_path_str)
    stem = video_path.stem

    if cache_path.exists():
        return stem, cache_path_str  # already cached

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return stem, ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    extractor = MediaPipeHandsExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )

    frame_ids: list[int] = []
    valid_flags: list[bool] = []
    landmarks_list: list[np.ndarray] = []
    detect_confs: list[float] = []
    track_confs: list[float] = []
    handedness_list: list[str | None] = []

    frame_id = 1  # 1-indexed to match IPN annotations
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_id - 1) * 1000.0 / fps)
        packet = FramePacket(frame=bgr, timestamp_ms=timestamp_ms, frame_id=frame_id)
        lp = extractor.extract(packet)

        frame_ids.append(frame_id)
        valid_flags.append(lp.valid)
        detect_confs.append(float(lp.detect_conf))
        track_confs.append(float(lp.track_conf))
        handedness_list.append(lp.handedness)

        if lp.valid and len(lp.landmarks) == 21:
            landmarks_list.append(np.asarray(lp.landmarks, dtype=np.float32))
        else:
            landmarks_list.append(np.zeros((21, 3), dtype=np.float32))

        frame_id += 1

    cap.release()
    extractor.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fps=np.float32(fps),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        valid=np.asarray(valid_flags, dtype=bool),
        landmarks=np.asarray(landmarks_list, dtype=np.float32),
        detect_conf=np.asarray(detect_confs, dtype=np.float32),
        track_conf=np.asarray(track_confs, dtype=np.float32),
        handedness=np.asarray(handedness_list, dtype=object),
    )
    logger.info("Cached %d frames → %s", len(frame_ids), cache_path.name)
    return stem, cache_path_str


def _load_cache(cache_path: Path) -> VideoLandmarks:
    with np.load(cache_path, allow_pickle=True) as data:
        return VideoLandmarks(
            video_stem=cache_path.stem,
            fps=float(data["fps"]),
            frame_ids=data["frame_ids"],
            valid=data["valid"],
            landmarks=data["landmarks"],
            detect_conf=data["detect_conf"],
            track_conf=data["track_conf"],
            handedness=list(data["handedness"]),
        )


def extract_all_landmarks(
    video_dir: Path,
    video_stems: list[str],
    cache_dir: Path,
    *,
    num_workers: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    no_cache: bool,
) -> dict[str, Path]:
    """Extract or load cached landmarks for every requested video.

    Returns a dict mapping video stem → cache NPZ path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, float, float]] = []
    result_paths: dict[str, Path] = {}

    for stem in video_stems:
        video_path = video_dir / f"{stem}.avi"
        if not video_path.exists():
            logger.warning("Video not found, skipping: %s", video_path)
            continue

        cache_path = cache_dir / f"{stem}.npz"
        if no_cache and cache_path.exists():
            cache_path.unlink()

        tasks.append(
            (
                str(video_path),
                str(cache_path),
                min_detection_confidence,
                min_tracking_confidence,
            )
        )

    if not tasks:
        logger.warning("No videos to process.")
        return {}

    logger.info("Processing %d videos with %d worker(s)…", len(tasks), num_workers)
    t0 = time.perf_counter()

    if num_workers <= 1:
        for task in tasks:
            stem, cpath = _extract_video_worker(task)
            if cpath:
                result_paths[stem] = Path(cpath)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_video_worker, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                stem, cpath = future.result()
                done += 1
                if cpath:
                    result_paths[stem] = Path(cpath)
                    logger.info("[%d/%d] %s", done, len(tasks), stem)
                else:
                    logger.warning("[%d/%d] FAILED: %s", done, len(tasks), stem)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Landmark extraction complete in %.1fs (%d videos cached).",
        elapsed,
        len(result_paths),
    )
    return result_paths


# ---------------------------------------------------------------------------
# Phase 2 — dataset assembly
# ---------------------------------------------------------------------------


def parse_annotations(annot_path: Path) -> list[AnnotationRow]:
    """Parse an IPN annotation CSV into a list of AnnotationRow objects."""
    rows: list[AnnotationRow] = []
    with annot_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            rows.append(
                AnnotationRow(
                    video=record["video"].strip(),
                    ipn_label=record["label"].strip(),
                    label_id=int(record["id"]),
                    t_start=int(record["t_start"]),
                    t_end=int(record["t_end"]),
                    n_frames=int(record["frames"]),
                )
            )
    return rows


def _group_annotations_by_video(
    rows: list[AnnotationRow],
) -> dict[str, list[AnnotationRow]]:
    grouped: dict[str, list[AnnotationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.video].append(row)
    # Sort each video's segments by start frame for sequential processing.
    for segs in grouped.values():
        segs.sort(key=lambda r: r.t_start)
    return dict(grouped)


def _subsample_unknown_segments(
    segments: list[AnnotationRow],
    max_per_source: int,
    rng: random.Random,
) -> list[AnnotationRow]:
    """Randomly subsample unknown-mapped segments per IPN source label."""
    by_source: dict[str, list[AnnotationRow]] = defaultdict(list)
    other: list[AnnotationRow] = []
    for seg in segments:
        if IPN_TO_VOCAB.get(seg.ipn_label) == "unknown":
            by_source[seg.ipn_label].append(seg)
        else:
            other.append(seg)

    subsampled: list[AnnotationRow] = list(other)
    for source_label, source_segs in by_source.items():
        if len(source_segs) > max_per_source:
            chosen = rng.sample(source_segs, max_per_source)
            logger.info(
                "Subsampled unknown source '%s': %d → %d segments.",
                source_label,
                len(source_segs),
                max_per_source,
            )
        else:
            chosen = source_segs
        subsampled.extend(chosen)
    return subsampled


def assemble_dataset(
    annotations: list[AnnotationRow],
    cache_paths: dict[str, Path],
    vocabulary: list[str],
    *,
    window_length: int,
    window_step: int,
    min_valid_ratio: float,
    label_strategy: str,
    max_unknown_segments_per_source: int,
    max_unknown_frames_per_segment: int,
    enable_orientation_normalization: bool,
    seed: int,
) -> SequenceDatasetBuilder:
    """Feed annotated landmark segments into a ``SequenceDatasetBuilder``.

    Each annotated gesture segment is assigned a unique ``recording_id`` of
    the form ``{video}:{seg_index}`` so the builder resets its feature
    transformer state at every segment boundary, ensuring velocity features
    are never computed across gesture boundaries.
    """
    rng = random.Random(seed)

    builder = SequenceDatasetBuilder(
        vocabulary=vocabulary,
        window_length=window_length,
        window_step=window_step,
        min_valid_ratio=min_valid_ratio,
        label_strategy=label_strategy,
        enable_orientation_normalization=enable_orientation_normalization,
    )

    # Group by video and apply unknown subsampling at the whole-dataset level.
    all_mapped: list[tuple[str, AnnotationRow]] = []  # (video_stem, row)
    for row in annotations:
        if row.ipn_label not in IPN_TO_VOCAB:
            continue  # skip G01, G02, G08, G09
        all_mapped.append((row.video, row))

    # Subsample unknown segments from the full list.
    subsampled_unknown = _subsample_unknown_segments(
        [r for _, r in all_mapped if IPN_TO_VOCAB[r.ipn_label] == "unknown"],
        max_unknown_segments_per_source,
        rng,
    )
    unknown_set = {id(r) for r in subsampled_unknown}

    # Rebuild final segment list preserving order and video grouping.
    final_by_video: dict[str, list[AnnotationRow]] = defaultdict(list)
    for video, row in all_mapped:
        if IPN_TO_VOCAB[row.ipn_label] == "unknown":
            if id(row) not in unknown_set:
                continue
        final_by_video[video].append(row)

    # Sort each video's segments by t_start for sequential frame access.
    for segs in final_by_video.values():
        segs.sort(key=lambda r: r.t_start)

    total_windows = 0
    missing_cache = 0

    for video_stem, segments in sorted(final_by_video.items()):
        cache_path = cache_paths.get(video_stem)
        if cache_path is None or not cache_path.exists():
            logger.warning("No cache for video '%s', skipping.", video_stem)
            missing_cache += 1
            continue

        vl = _load_cache(cache_path)
        # Build a fast frame_id → index lookup.
        fid_to_idx = {int(fid): i for i, fid in enumerate(vl.frame_ids)}

        for seg_idx, seg in enumerate(segments):
            vocab_label = IPN_TO_VOCAB[seg.ipn_label]
            recording_id = f"{video_stem}:seg{seg_idx:04d}"

            # Iterate frames t_start..t_end (1-indexed inclusive).
            # Cap unknown segments to avoid class imbalance from long background
            # spans generating far more windows than short gesture segments.
            t_end_effective = seg.t_end
            if vocab_label == "unknown" and max_unknown_frames_per_segment > 0:
                t_end_effective = min(
                    seg.t_end, seg.t_start + max_unknown_frames_per_segment - 1
                )

            seg_windows_before = len(builder.samples)
            for frame_id in range(seg.t_start, t_end_effective + 1):
                idx = fid_to_idx.get(frame_id)
                if idx is None:
                    continue  # frame not in cache (truncated video)

                timestamp_ms = int((frame_id - 1) * 1000.0 / vl.fps)
                is_valid = bool(vl.valid[idx])
                lm = vl.landmarks[idx].tolist()  # List[List[float]] shape (21, 3)

                lf = LabeledLandmarkFrame(
                    recording_id=recording_id,
                    timestamp_ms=timestamp_ms,
                    label=vocab_label,
                    landmarks=lm,
                    valid=is_valid,
                    handedness=vl.handedness[idx],
                    detect_conf=float(vl.detect_conf[idx]),
                    track_conf=float(vl.track_conf[idx]),
                )
                builder.add_frame(lf)

            new_windows = len(builder.samples) - seg_windows_before
            total_windows += new_windows

    if missing_cache > 0:
        logger.warning("%d video(s) had no cache and were skipped.", missing_cache)

    logger.info(
        "Assembly complete: %d sliding-window samples generated.", total_windows
    )
    return builder


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _compute_class_stats(builder: SequenceDatasetBuilder) -> dict[str, Any]:
    from collections import Counter

    label_counts = Counter(s.label for s in builder.samples)
    return {
        "total_samples": len(builder.samples),
        "class_distribution": dict(label_counts),
        "num_classes": len(label_counts),
    }


def _load_vocabulary(vocab_config_path: Path) -> list[str]:
    with vocab_config_path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    gestures = payload.get("gestures", [])
    if not isinstance(gestures, list) or not gestures:
        raise ValueError(
            f"Vocabulary config '{vocab_config_path}' must contain a non-empty"
            " 'gestures' list."
        )
    return [str(g) for g in gestures]


def _save_metadata(
    metadata_path: Path,
    annotations_used: int,
    cache_paths: dict[str, Path],
    vocabulary: list[str],
    window_length: int,
    window_step: int,
    stats: dict[str, Any],
) -> None:
    metadata = {
        "ipn_label_mapping": IPN_TO_VOCAB,
        "vocabulary": vocabulary,
        "window_length": window_length,
        "window_step": window_step,
        "feature_schema_version": "feature-schema-v1",
        "annotation_rows_used": annotations_used,
        "videos_processed": len(cache_paths),
        "dataset_stats": stats,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written to %s", metadata_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ipn-root",
        default="data/IPN",
        help="Root directory of the IPN Hand dataset (default: data/IPN).",
    )
    p.add_argument(
        "--output-npz",
        required=True,
        help="Output path for the generated sequence dataset NPZ.",
    )
    p.add_argument(
        "--annotation-file",
        default="Annot_List.txt",
        help=(
            "Annotation filename inside <ipn-root>/annotations/ "
            "(default: Annot_List.txt).  Use Annot_TrainList.txt or "
            "Annot_TestList.txt to follow the original IPN splits."
        ),
    )
    p.add_argument(
        "--vocab-config",
        default="configs/gestures/vocabulary_ipn.yaml",
        help="Path to vocabulary YAML (default: configs/gestures/vocabulary_ipn.yaml).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for landmark extraction.  "
            "Set to 0 to use all CPU cores. (default: 1)"
        ),
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing per-video caches and re-extract all landmarks.",
    )
    p.add_argument(
        "--max-unknown-segments-per-source",
        type=int,
        default=150,
        metavar="N",
        help=(
            "Maximum number of unknown-class segments drawn from each IPN "
            "source label (D0X, B0A, B0B) to limit class imbalance.  "
            "Default: 150."
        ),
    )
    p.add_argument(
        "--max-unknown-frames-per-segment",
        type=int,
        default=60,
        metavar="F",
        help=(
            "Maximum frames consumed from each unknown-class segment.  "
            "Long background spans (D0X) can otherwise generate hundreds of "
            "windows per segment.  Set to 0 to disable the cap.  Default: 60."
        ),
    )
    p.add_argument(
        "--window-length",
        type=int,
        default=24,
        help="Sliding window length in frames (default: 24).",
    )
    p.add_argument(
        "--window-step",
        type=int,
        default=3,
        help="Sliding window step size in frames (default: 3).",
    )
    p.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.8,
        help="Minimum ratio of valid frames required to emit a window (default: 0.8).",
    )
    p.add_argument(
        "--label-strategy",
        default="majority",
        choices=["last", "majority"],
        help=(
            "Label assignment strategy for sequence windows.  'majority' is "
            "recommended for IPN since all frames in a segment share the same "
            "label. (default: majority)"
        ),
    )
    p.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe minimum detection confidence (default: 0.7).",
    )
    p.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence (default: 0.5).",
    )
    p.add_argument(
        "--enable-orientation-normalization",
        action="store_true",
        help="Enable orientation normalization in the feature transformer.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for unknown-segment subsampling (default: 42).",
    )
    p.add_argument(
        "--metadata-json",
        default=None,
        help=(
            "Optional path to write a JSON metadata summary alongside the NPZ.  "
            "Defaults to <output-npz>.metadata.json when not specified."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ipn_root = Path(args.ipn_root)
    video_dir = ipn_root / "videos"
    annot_path = ipn_root / "annotations" / args.annotation_file
    cache_dir = ipn_root / "cache"
    output_npz = Path(args.output_npz)
    vocab_config = Path(args.vocab_config)

    # Validate inputs.
    if not video_dir.is_dir():
        parser.error(f"Video directory not found: {video_dir}")
    if not annot_path.is_file():
        parser.error(f"Annotation file not found: {annot_path}")
    if not vocab_config.is_file():
        parser.error(f"Vocabulary config not found: {vocab_config}")

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    # -----------------------------------------------------------------------
    # Load vocabulary and annotations.
    # -----------------------------------------------------------------------
    vocabulary = _load_vocabulary(vocab_config)
    logger.info("Vocabulary (%d classes): %s", len(vocabulary), vocabulary)

    annotations = parse_annotations(annot_path)
    logger.info("Loaded %d annotation rows from %s.", len(annotations), annot_path.name)

    # Collect the unique video stems that appear in (mapped) annotations.
    needed_videos: list[str] = sorted(
        {row.video for row in annotations if row.ipn_label in IPN_TO_VOCAB}
    )
    logger.info("%d unique videos needed for mapped labels.", len(needed_videos))

    # -----------------------------------------------------------------------
    # Phase 1 — extract / load per-video landmark caches.
    # -----------------------------------------------------------------------
    cache_paths = extract_all_landmarks(
        video_dir=video_dir,
        video_stems=needed_videos,
        cache_dir=cache_dir,
        num_workers=num_workers,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        no_cache=args.no_cache,
    )
    if not cache_paths:
        logger.error("No videos were processed successfully.  Aborting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 2 — assemble sliding-window dataset.
    # -----------------------------------------------------------------------
    builder = assemble_dataset(
        annotations=annotations,
        cache_paths=cache_paths,
        vocabulary=vocabulary,
        window_length=args.window_length,
        window_step=args.window_step,
        min_valid_ratio=args.min_valid_ratio,
        label_strategy=args.label_strategy,
        max_unknown_segments_per_source=args.max_unknown_segments_per_source,
        max_unknown_frames_per_segment=args.max_unknown_frames_per_segment,
        enable_orientation_normalization=args.enable_orientation_normalization,
        seed=args.seed,
    )

    if not builder.samples:
        logger.error(
            "No sequence samples were generated.  Check your annotations, "
            "vocabulary, and window settings."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Save NPZ and metadata.
    # -----------------------------------------------------------------------
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    builder.save_npz(output_npz)
    logger.info("Dataset saved → %s  (%d samples)", output_npz, len(builder.samples))

    stats = _compute_class_stats(builder)
    logger.info("Class distribution: %s", stats["class_distribution"])

    metadata_path = Path(
        args.metadata_json if args.metadata_json else str(output_npz) + ".metadata.json"
    )
    _save_metadata(
        metadata_path=metadata_path,
        annotations_used=len(annotations),
        cache_paths=cache_paths,
        vocabulary=vocabulary,
        window_length=args.window_length,
        window_step=args.window_step,
        stats=stats,
    )


if __name__ == "__main__":
    main()
