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
        The ``unknown`` class (D0X) is subsampled to prevent it from
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
to ``--dataset-path``.
The vocabulary is embedded in the NPZ and does not need to be passed separately.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:  # graceful degradation if tqdm is not installed

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a plain script from any cwd.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maestro.training.dataset_builder import (  # noqa: E402
    LabeledLandmarkFrame,
    SequenceDatasetBuilder,
)

try:  # noqa: E402
    from maestro.domain.entities import LabeledGraphFrame  # type: ignore
except ImportError:  # graph-mode legacy type is optional in this repo state
    LabeledGraphFrame = None  # type: ignore[assignment]

try:  # noqa: E402
    from maestro.training.dataset_builder import GraphSequenceDatasetBuilder  # type: ignore
except ImportError:  # graph-mode builder may not exist; flat mode remains supported
    GraphSequenceDatasetBuilder = None  # type: ignore[assignment]


class _GraphSchemaCompat:
    GRAPH_SCHEMA_VERSION = "graph-schema-v1"
    NUM_NODES = 67
    NUM_COORDS = 3
    NODE_INDEX_MAP = {
        "right_hand": (0, 21),
        "left_hand": (21, 42),
        "pose": (42, 53),
        "face": (53, 67),
    }
    POSE_LANDMARK_INDICES = list(range(11))
    FACE_LANDMARK_INDICES = list(range(14))


graph_schema = _GraphSchemaCompat()

# Default model cache path (mirrors the constant formerly in holistic_extractor).
_DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "maestro" / "holistic_landmarker.task"


def _is_nan(lm: object) -> bool:
    """Return True if any coordinate of *lm* is NaN."""
    import math

    return (
        math.isnan(getattr(lm, "x", 0.0))
        or math.isnan(getattr(lm, "y", 0.0))
        or math.isnan(getattr(lm, "z", 0.0))
    )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPN label → production vocabulary mapping
# ---------------------------------------------------------------------------
# Full IPN baseline mapping (14 classes):
#   D0X, B0A, B0B, G01..G11

IPN_TO_VOCAB: dict[str, str] = {
    "D0X": "unknown",  # non-gesture background
    "B0A": "point_one",  # pointing with one finger
    "B0B": "point_two",  # pointing with two fingers
    "G03": "swipe_up",  # throw up
    "G04": "swipe_down",  # throw down
    "G05": "swipe_left",  # throw left
    "G06": "swipe_right",  # throw right
    "G07": "open_palm_hold",  # open twice
    "G10": "zoom_in",  # zoom in
    "G11": "zoom_out",  # zoom out
    "G01": "click_one",  # click with one finger
    "G02": "click_two",  # click with two fingers
    "G08": "double_click_one",  # double click with one finger
    "G09": "double_click_two",  # double click with two fingers
}

# Binary mapping for detector training (Phase 3 — mirrors ipnbinary.json):
#   no_gesture (0) ← D0X non-gesture background
#   gesture    (1) ← all 13 IPN gesture classes
IPN_TO_VOCAB_BINARY: dict[str, str] = {
    "D0X": "no_gesture",
    "B0A": "gesture",
    "B0B": "gesture",
    "G01": "gesture",
    "G02": "gesture",
    "G03": "gesture",
    "G04": "gesture",
    "G05": "gesture",
    "G06": "gesture",
    "G07": "gesture",
    "G08": "gesture",
    "G09": "gesture",
    "G10": "gesture",
    "G11": "gesture",
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
    """Per-frame holistic landmark results for a single video (graph-schema-v1)."""

    video_stem: str
    fps: float
    frame_ids: np.ndarray  # shape (N,)     int32, 1-indexed
    keypoints: np.ndarray  # shape (N, 67, 3) float32 — raw (pre-normalisation)
    valid_mask: np.ndarray  # shape (N, 67)  bool — per-node detection validity


# ---------------------------------------------------------------------------
# Phase 1 — per-video landmark extraction (worker)
# ---------------------------------------------------------------------------


def _parse_holistic_legacy(
    result: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a legacy mp.solutions.holistic result into (67, 3) keypoints + mask.

    The legacy solutions API returns ``NormalizedLandmarkList`` objects that
    expose landmarks via a ``.landmark`` attribute, unlike the task-based API
    which returns plain lists.  No normalisation is applied — raw coordinates
    are stored in the per-video cache; ``graph_schema.normalise_graph_frame()``
    is applied at assembly time.
    """
    keypoints = np.zeros((graph_schema.NUM_NODES, graph_schema.NUM_COORDS), dtype=np.float32)
    valid_mask = np.zeros(graph_schema.NUM_NODES, dtype=bool)

    rh = result.right_hand_landmarks
    if rh:
        for i, lm in enumerate(rh.landmark[:21]):
            if not _is_nan(lm):
                keypoints[i] = [lm.x, lm.y, lm.z]
                valid_mask[i] = True

    lh_offset = graph_schema.NODE_INDEX_MAP["left_hand"][0]
    lh = result.left_hand_landmarks
    if lh:
        for i, lm in enumerate(lh.landmark[:21]):
            idx = i + lh_offset
            if not _is_nan(lm):
                keypoints[idx] = [lm.x, lm.y, lm.z]
                valid_mask[idx] = True

    pose = result.pose_landmarks
    if pose:
        for gi, mp_i in enumerate(graph_schema.POSE_LANDMARK_INDICES):
            idx = graph_schema.NODE_INDEX_MAP["pose"][0] + gi
            if mp_i < len(pose.landmark):
                lm = pose.landmark[mp_i]
                if not _is_nan(lm):
                    keypoints[idx] = [lm.x, lm.y, lm.z]
                    valid_mask[idx] = True

    face = result.face_landmarks
    if face:
        for gi, mp_i in enumerate(graph_schema.FACE_LANDMARK_INDICES):
            idx = graph_schema.NODE_INDEX_MAP["face"][0] + gi
            if mp_i < len(face.landmark):
                lm = face.landmark[mp_i]
                if not _is_nan(lm):
                    keypoints[idx] = [lm.x, lm.y, lm.z]
                    valid_mask[idx] = True

    return keypoints, valid_mask


def _extract_video_worker(args: tuple[str, str, str, float, float]) -> tuple[str, str]:
    """Worker function: extract holistic landmarks for a single video.

    Parameters
    ----------
    args:
        Tuple of (video_path, cache_path, model_path,
        min_detection_confidence, min_tracking_confidence).

    Returns
    -------
    (video_stem, cache_path)
        On failure the stem is returned with an empty cache_path string.
    """
    import mediapipe as mp

    video_path_str, cache_path_str, model_path_str, min_det_conf, min_track_conf = args
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

    # Use the legacy mp.solutions.holistic.Holistic API for offline extraction.
    # The task-based HolisticLandmarker (mp.tasks.vision) triggers a fatal
    # SIGABRT on CPU-only headless Linux due to an assertion in the inference
    # feedback manager when EGL/GPU is unavailable.  The legacy solutions API
    # does not use that mechanism and is stable for offline dataset building.
    frame_ids: list[int] = []
    keypoints_list: list[np.ndarray] = []
    valid_mask_list: list[np.ndarray] = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
        enable_segmentation=False,
    ) as holistic:
        frame_id = 1  # 1-indexed to match IPN annotations
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            kp, vm = _parse_holistic_legacy(result)
            frame_ids.append(frame_id)
            keypoints_list.append(kp)
            valid_mask_list.append(vm)
            frame_id += 1

    cap.release()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fps=np.float32(fps),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        keypoints=np.asarray(keypoints_list, dtype=np.float32),
        valid_mask=np.asarray(valid_mask_list, dtype=bool),
    )
    logger.info("Cached %d frames → %s", len(frame_ids), cache_path.name)
    return stem, cache_path_str


def _load_cache(cache_path: Path) -> VideoLandmarks:
    """Load a holistic or legacy right-hand-only cache NPZ.

    Holistic format (``*_holistic.npz`` written by this script):
        ``keypoints``  (N, 67, 3) float32 — all holistic nodes
        ``valid_mask`` (N, 67)    bool    — per-node validity

    Legacy format (``{stem}.npz`` written by earlier pipeline versions):
        ``landmarks``  (N, 21, 3) float32 — right-hand landmarks only
        ``valid``      (N,)       bool    — per-frame detection validity

    Legacy caches are promoted to the holistic layout by placing the 21
    right-hand landmarks at nodes 0-20 (the right-hand slot in the graph
    schema), leaving nodes 21-66 as zeros.  This is sufficient because the
    flat pipeline only reads ``keypoints[:, :21, :]``.
    """
    with np.load(cache_path, allow_pickle=True) as data:
        fps = float(data["fps"])
        frame_ids = data["frame_ids"]

        if "keypoints" in data.files:
            # Current holistic format — load directly.
            return VideoLandmarks(
                video_stem=cache_path.stem,
                fps=fps,
                frame_ids=frame_ids,
                keypoints=data["keypoints"],  # (N, 67, 3)
                valid_mask=data["valid_mask"],  # (N, 67)
            )

        # Legacy format: landmarks (N, 21, 3) + valid (N,)
        rh_landmarks = data["landmarks"]  # (N, 21, 3) right-hand only
        rh_valid = data["valid"]  # (N,) bool
        N = len(frame_ids)

        # Promote to full holistic layout (67 nodes, 3 coords) so the rest
        # of the pipeline can use the same code path regardless of format.
        keypoints = np.zeros((N, graph_schema.NUM_NODES, graph_schema.NUM_COORDS), dtype=np.float32)
        keypoints[:, :21, :] = rh_landmarks

        valid_mask = np.zeros((N, graph_schema.NUM_NODES), dtype=bool)
        valid_mask[:, :21] = rh_valid[:, np.newaxis]

        return VideoLandmarks(
            video_stem=cache_path.stem,
            fps=fps,
            frame_ids=frame_ids,
            keypoints=keypoints,
            valid_mask=valid_mask,
        )


def extract_all_landmarks(
    video_dir: Path,
    video_stems: list[str],
    cache_dir: Path,
    *,
    num_workers: int,
    model_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    no_cache: bool,
) -> dict[str, Path]:
    """Extract or load cached holistic landmarks for every requested video.

    Returns a dict mapping video stem → cache NPZ path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, str, float, float]] = []
    result_paths: dict[str, Path] = {}

    for stem in video_stems:
        cache_path = cache_dir / f"{stem}_holistic.npz"

        # Fast path: use an existing cache file without requiring the video.
        # This allows Phase 2 to proceed even when the original .avi files
        # have been moved or deleted after the first extraction run.
        if not no_cache and cache_path.exists():
            result_paths[stem] = cache_path
            continue

        # Legacy fallback: previous versions of this script saved caches as
        # {stem}.npz (without the _holistic suffix).  Accept those files so
        # that users who have already extracted landmarks do not need to
        # re-run Phase 1.
        legacy_cache_path = cache_dir / f"{stem}.npz"
        if not no_cache and legacy_cache_path.exists():
            logger.debug("Using legacy cache for %s (%s)", stem, legacy_cache_path.name)
            result_paths[stem] = legacy_cache_path
            continue

        # Cache miss — queue the video for (re-)extraction.
        video_path = video_dir / f"{stem}.avi"
        if not video_path.exists():
            logger.warning("Video not found and no cache available, skipping: %s", video_path)
            continue

        if no_cache and cache_path.exists():
            cache_path.unlink()

        tasks.append(
            (
                str(video_path),
                str(cache_path),
                model_path,
                min_detection_confidence,
                min_tracking_confidence,
            )
        )

    if result_paths:
        logger.info(
            "%d video(s) resolved from existing cache; %d video(s) queued for extraction.",
            len(result_paths),
            len(tasks),
        )

    if not tasks and not result_paths:
        logger.warning("No videos to process and no existing caches found.")
        return {}

    logger.info("Processing %d videos with %d worker(s)…", len(tasks), num_workers)
    t0 = time.perf_counter()

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Extracting landmarks", unit="video"):
            stem, cpath = _extract_video_worker(task)
            if cpath:
                result_paths[stem] = Path(cpath)
    else:
        chunksize = max(1, min(32, len(tasks) // max(1, num_workers * 8)))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            iterator = executor.map(_extract_video_worker, tasks, chunksize=chunksize)
            for stem, cpath in tqdm(
                iterator, total=len(tasks), desc="Extracting landmarks", unit="video"
            ):
                if cpath:
                    result_paths[stem] = Path(cpath)
                else:
                    logger.warning("Extraction failed: %s", stem)

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
    """Randomly subsample unknown-mapped segments per source video.

    Groups D0X segments by video stem so that each recording contributes at
    most ``max_per_source`` unknown segments.  Grouping by IPN label alone
    (e.g. a single 'D0X' bucket for all 200 videos) would keep only
    ``max_per_source`` segments globally, collapsing coverage to a handful
    of videos.
    """
    by_video: dict[str, list[AnnotationRow]] = defaultdict(list)
    other: list[AnnotationRow] = []
    for seg in segments:
        if IPN_TO_VOCAB.get(seg.ipn_label) == "unknown":
            by_video[seg.video].append(seg)
        else:
            other.append(seg)

    subsampled: list[AnnotationRow] = list(other)
    for video_stem, video_segs in by_video.items():
        if len(video_segs) > max_per_source:
            chosen = rng.sample(video_segs, max_per_source)
            logger.debug(
                "Subsampled unknown for video '%s': %d → %d segments.",
                video_stem,
                len(video_segs),
                max_per_source,
            )
        else:
            chosen = video_segs
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
    seed: int,
    label_mapping: dict[str, str] | None = None,
) -> GraphSequenceDatasetBuilder:
    """Feed annotated holistic frames into a ``GraphSequenceDatasetBuilder``.

    Each annotated gesture segment is assigned a unique ``recording_id`` of
    the form ``{video}:{seg_index}`` so the builder resets its buffer state
    at every segment boundary.
    """
    if GraphSequenceDatasetBuilder is None or LabeledGraphFrame is None:
        raise RuntimeError(
            "Graph dataset format is not available in this codebase state. "
            "Use '--format flat' (recommended for current training pipeline)."
        )

    effective_mapping = label_mapping if label_mapping is not None else IPN_TO_VOCAB
    # For binary mode the "background" class may be called 'no_gesture' not 'unknown'.
    # Determine the background label name so unknown-subsampling logic can be skipped.
    _is_binary = label_mapping is not None

    rng = random.Random(seed)

    builder = GraphSequenceDatasetBuilder(
        vocabulary=vocabulary,
        window_length=window_length,
        window_step=window_step,
        min_valid_ratio=min_valid_ratio,
        label_strategy=label_strategy,
    )

    # Group by video and apply unknown subsampling at the whole-dataset level.
    all_mapped: list[tuple[str, AnnotationRow]] = []  # (video_stem, row)
    vocabulary_set = set(vocabulary)
    for row in annotations:
        if row.ipn_label not in effective_mapping:
            continue  # IPN code has no production mapping at all
        if effective_mapping[row.ipn_label] not in vocabulary_set:
            continue  # mapped label not in target vocabulary (e.g. click_* with 8-class vocab)
        all_mapped.append((row.video, row))

    # Subsample unknown segments from the full list (skipped in binary mode —
    # both classes are equally wanted for detector training).
    if not _is_binary:
        subsampled_unknown = _subsample_unknown_segments(
            [r for _, r in all_mapped if effective_mapping[r.ipn_label] == "unknown"],
            max_unknown_segments_per_source,
            rng,
        )
        unknown_set = {id(r) for r in subsampled_unknown}
    else:
        unknown_set: set[int] = set()

    # Rebuild final segment list preserving order and video grouping.
    final_by_video: dict[str, list[AnnotationRow]] = defaultdict(list)
    for video, row in all_mapped:
        if not _is_binary and effective_mapping[row.ipn_label] == "unknown":
            if id(row) not in unknown_set:
                continue
        final_by_video[video].append(row)

    # Sort each video's segments by t_start for sequential frame access.
    for segs in final_by_video.values():
        segs.sort(key=lambda r: r.t_start)

    total_windows = 0
    missing_cache = 0

    video_items = sorted(final_by_video.items())
    for video_stem, segments in tqdm(video_items, desc="Assembling dataset", unit="video"):
        cache_path = cache_paths.get(video_stem)
        if cache_path is None or not cache_path.exists():
            logger.warning("No cache for video '%s', skipping.", video_stem)
            missing_cache += 1
            continue

        vl = _load_cache(cache_path)
        # Build a fast frame_id → index lookup.
        fid_to_idx = {int(fid): i for i, fid in enumerate(vl.frame_ids)}

        for seg_idx, seg in enumerate(segments):
            vocab_label = effective_mapping[seg.ipn_label]
            recording_id = f"{video_stem}:seg{seg_idx:04d}"

            # Iterate frames t_start..t_end (1-indexed inclusive).
            # Cap unknown segments to avoid class imbalance from long background
            # spans generating far more windows than short gesture segments.
            # In binary mode this cap is disabled — the no_gesture class is needed.
            t_end_effective = seg.t_end
            if not _is_binary and vocab_label == "unknown" and max_unknown_frames_per_segment > 0:
                t_end_effective = min(seg.t_end, seg.t_start + max_unknown_frames_per_segment - 1)

            seg_windows_before = len(builder.samples)
            for frame_id in range(seg.t_start, t_end_effective + 1):
                idx = fid_to_idx.get(frame_id)
                if idx is None:
                    continue  # frame not in cache (truncated video)

                timestamp_ms = int((frame_id - 1) * 1000.0 / vl.fps)

                lf = LabeledGraphFrame(
                    keypoints=vl.keypoints[idx],  # (67, 3) raw
                    valid_mask=vl.valid_mask[idx],  # (67,) bool
                    label=vocab_label,
                    recording_id=recording_id,
                    timestamp_ms=timestamp_ms,
                )
                builder.add_frame(lf)

            new_windows = len(builder.samples) - seg_windows_before
            total_windows += new_windows

    if missing_cache > 0:
        logger.warning("%d video(s) had no cache and were skipped.", missing_cache)

    logger.info("Assembly complete: %d sliding-window samples generated.", total_windows)
    return builder


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _compute_class_stats(
    builder: GraphSequenceDatasetBuilder | SequenceDatasetBuilder,
) -> dict[str, Any]:
    from collections import Counter

    label_counts = Counter(s.label for s in builder.samples)
    return {
        "total_samples": len(builder.samples),
        "class_distribution": dict(label_counts),
        "num_classes": len(label_counts),
    }


def assemble_flat_dataset(
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
    seed: int,
    label_mapping: dict[str, str] | None = None,
) -> SequenceDatasetBuilder:
    """Feed annotated right-hand landmarks into a ``SequenceDatasetBuilder``.

    Extracts right-hand keypoints (nodes 0-20) from the holistic cache and
    produces a flat feature-schema-v5 dataset compatible with
    ``GestureDataModule`` and ``TwoStreamGestureLSTM``.
    """
    effective_mapping = label_mapping if label_mapping is not None else IPN_TO_VOCAB
    _is_binary = label_mapping is not None

    rng = random.Random(seed)

    builder = SequenceDatasetBuilder(
        vocabulary=vocabulary,
        window_length=window_length,
        window_step=window_step,
        min_valid_ratio=min_valid_ratio,
        label_strategy=label_strategy,
        spill_sequences_to_disk=True,
    )

    all_mapped: list[tuple[str, AnnotationRow]] = []
    vocabulary_set = set(vocabulary)
    for row in annotations:
        if row.ipn_label not in effective_mapping:
            continue  # IPN code has no production mapping at all
        if effective_mapping[row.ipn_label] not in vocabulary_set:
            continue  # mapped label not in target vocabulary (e.g. click_* with 8-class vocab)
        all_mapped.append((row.video, row))

    if not _is_binary:
        subsampled_unknown = _subsample_unknown_segments(
            [r for _, r in all_mapped if effective_mapping[r.ipn_label] == "unknown"],
            max_unknown_segments_per_source,
            rng,
        )
        unknown_set = {id(r) for r in subsampled_unknown}
    else:
        unknown_set: set[int] = set()

    final_by_video: dict[str, list[AnnotationRow]] = defaultdict(list)
    for video, row in all_mapped:
        if not _is_binary and effective_mapping[row.ipn_label] == "unknown":
            if id(row) not in unknown_set:
                continue
        final_by_video[video].append(row)

    for segs in final_by_video.values():
        segs.sort(key=lambda r: r.t_start)

    total_windows = 0
    missing_cache = 0

    video_items = sorted(final_by_video.items())
    for video_stem, segments in tqdm(video_items, desc="Assembling dataset", unit="video"):
        cache_path = cache_paths.get(video_stem)
        if cache_path is None or not cache_path.exists():
            logger.warning("No cache for video '%s', skipping.", video_stem)
            missing_cache += 1
            continue

        vl = _load_cache(cache_path)
        fid_to_idx = {int(fid): i for i, fid in enumerate(vl.frame_ids)}

        for seg_idx, seg in enumerate(segments):
            vocab_label = effective_mapping[seg.ipn_label]
            recording_id = f"{video_stem}:seg{seg_idx:04d}"

            t_end_effective = seg.t_end
            if not _is_binary and vocab_label == "unknown" and max_unknown_frames_per_segment > 0:
                t_end_effective = min(seg.t_end, seg.t_start + max_unknown_frames_per_segment - 1)

            seg_windows_before = len(builder.samples)
            for frame_id in range(seg.t_start, t_end_effective + 1):
                idx = fid_to_idx.get(frame_id)
                if idx is None:
                    continue

                timestamp_ms = int((frame_id - 1) * 1000.0 / vl.fps)
                # Extract right-hand keypoints (nodes 0-20 in graph layout).
                rh_kps = vl.keypoints[idx, :21]  # (21, 3) raw coordinates
                rh_valid = bool(vl.valid_mask[idx, :21].any())

                lf = LabeledLandmarkFrame(
                    recording_id=recording_id,
                    timestamp_ms=timestamp_ms,
                    label=vocab_label,
                    landmarks=rh_kps.tolist(),
                    valid=rh_valid,
                    handedness="Right",
                )
                builder.add_frame(lf)

            total_windows += len(builder.samples) - seg_windows_before

    if missing_cache > 0:
        logger.warning("%d video(s) had no cache and were skipped.", missing_cache)

    logger.info("Flat assembly complete: %d sliding-window samples generated.", total_windows)
    return builder


def _load_vocabulary(vocab_config_path: Path) -> list[str]:
    with vocab_config_path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    gestures = payload.get("gestures", [])
    if not isinstance(gestures, list) or not gestures:
        raise ValueError(
            f"Vocabulary config '{vocab_config_path}' must contain a non-empty 'gestures' list."
        )
    return [str(g) for g in gestures]


# ---------------------------------------------------------------------------
# Per-class chunk helpers (flat format)
# ---------------------------------------------------------------------------


def _save_ipn_chunks_streaming(
    annotations: list[AnnotationRow],
    cache_paths: dict[str, Path],
    vocabulary: list[str],
    chunk_dir: Path,
    *,
    window_length: int,
    window_step: int,
    min_valid_ratio: float,
    label_strategy: str,
    max_unknown_segments_per_source: int,
    max_unknown_frames_per_segment: int,
    seed: int,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Build one flat NPZ chunk per gesture class, processing one class at a time.

    Mirrors ``_save_chunks_streaming`` from ``build_jester_dataset.py`` but
    uses ``assemble_flat_dataset`` (right-hand 147-dim feature schema) and
    IPN's annotation structure.  Processing one class at a time bounds peak
    RAM to a single class rather than the full multi-class dataset.

    Parameters
    ----------
    skip_existing:
        If True (default), skip classes whose chunk file already exists and is
        valid.  Pass False to force a full rebuild.

    Returns
    -------
    dict mapping class_name → sample count for all saved chunks.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, int] = {}

    for cls in vocabulary:
        out_path = chunk_dir / f"{cls}.npz"
        if skip_existing and out_path.exists():
            try:
                with zipfile.ZipFile(out_path, "r") as _zf:
                    _zf.namelist()
                with np.load(out_path, allow_pickle=True) as raw:
                    n_existing = len(raw["labels"])
                logger.info("  ✓ %-25s (existing chunk, %d samples)", cls, n_existing)
                saved[cls] = n_existing
                continue
            except Exception:
                logger.warning("Corrupt chunk for %r — will rebuild: %s", cls, out_path.name)
                out_path.unlink(missing_ok=True)

        logger.info("  Building chunk: %-25s …", cls)
        cls_builder = assemble_flat_dataset(
            annotations=annotations,
            cache_paths=cache_paths,
            vocabulary=[cls],  # single-class vocabulary filters naturally
            window_length=window_length,
            window_step=window_step,
            min_valid_ratio=min_valid_ratio,
            label_strategy=label_strategy,
            max_unknown_segments_per_source=max_unknown_segments_per_source,
            max_unknown_frames_per_segment=max_unknown_frames_per_segment,
            seed=seed,
        )

        n = len(cls_builder.samples)
        if n == 0:
            logger.warning("Class %r yielded 0 windows — chunk not saved.", cls)
            del cls_builder
            continue

        # Write atomically via tempfile → os.replace.
        _tmp_fd, _tmp_str = tempfile.mkstemp(dir=chunk_dir, suffix=".npz")
        os.close(_tmp_fd)
        _tmp_path = Path(_tmp_str)
        try:
            cls_builder.save_npz(_tmp_path, feature_schema_version="feature-schema-v5")
            os.replace(_tmp_path, out_path)
        except Exception:
            _tmp_path.unlink(missing_ok=True)
            raise
        finally:
            del cls_builder
            gc.collect()  # Force release before next class build (point_one/point_two are large)

        size_mb = out_path.stat().st_size / 1024**2
        logger.info("    → %s  %d samples  %.1f MB", out_path.name, n, size_mb)
        saved[cls] = n

    total = sum(saved.values())
    logger.info(
        "Chunk build complete: %d classes  %d total samples  → %s",
        len(saved),
        total,
        chunk_dir,
    )
    return saved


def _assemble_ipn_from_chunks(
    chunk_dir: Path,
    vocabulary: list[str],
    output_path: Path,
    seed: int = 42,
) -> None:
    """Assemble a single IPN NPZ from pre-built per-class chunk files (fast path).

    Skips landmark extraction and dataset assembly entirely.  Loads the chunk
    file for each vocabulary class, concatenates, shuffles, and writes a single
    NPZ compatible with ``GestureDataModule``.

    Parameters
    ----------
    chunk_dir:
        Directory containing ``{class_name}.npz`` chunk files produced by a
        previous ``--chunk-dir`` run.
    vocabulary:
        Target vocabulary list (defines which chunk files to load and the
        canonical class ordering for ``label_ids``).
    output_path:
        Destination NPZ file.
    """
    logger.info(
        "Assembling %d-class IPN NPZ from chunks in %s …",
        len(vocabulary),
        chunk_dir,
    )
    label_to_id = {cls: idx for idx, cls in enumerate(vocabulary)}

    all_seqs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_label_ids: list[np.ndarray] = []
    all_rec_ids: list[np.ndarray] = []

    window_length: int | None = None

    for cls in vocabulary:
        chunk_path = chunk_dir / f"{cls}.npz"
        if not chunk_path.exists():
            logger.warning("Chunk missing for class %r: %s — skipped.", cls, chunk_path)
            continue
        with np.load(chunk_path, allow_pickle=True) as raw:
            seqs = raw["sequences"].astype(np.float32)
            if window_length is None:
                window_length = seqs.shape[1]
            all_seqs.append(seqs)
            all_labels.append(raw["labels"].astype(str))
            all_label_ids.append(np.full(len(seqs), label_to_id[cls], dtype=np.int64))
            all_rec_ids.append(raw["recording_ids"].astype(str))
        logger.info("  %-25s %6d samples", cls, len(seqs))

    if not all_seqs:
        logger.error("No chunk data loaded — aborting.")
        sys.exit(1)

    sequences = np.concatenate(all_seqs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    label_ids = np.concatenate(all_label_ids, axis=0)
    rec_ids = np.concatenate(all_rec_ids, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    sequences = sequences[perm]
    labels = labels[perm]
    label_ids = label_ids[perm]
    rec_ids = rec_ids[perm]

    dist = dict(Counter(labels.tolist()))
    logger.info(
        "Assembly complete: %d samples  %d classes",
        len(labels),
        len({c for c in vocabulary if dist.get(c, 0) > 0}),
    )
    for cls in vocabulary:
        logger.info("  %-25s %6d", cls, dist.get(cls, 0))

    metadata = {
        "feature_schema_version": "feature-schema-v5",
        "vocabulary": vocabulary,
        "num_classes": len(vocabulary),
        "window_length": int(window_length or 32),
        "assembled_from_chunks": str(chunk_dir),
        "seed": seed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sequences=sequences,
        labels=labels,
        label_ids=label_ids,
        recording_ids=rec_ids,
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
    size_mb = output_path.stat().st_size / 1024**2
    logger.info("Saved → %s  (%.1f MB)", output_path, size_mb)


def _save_metadata(
    metadata_path: Path,
    annotations_used: int,
    cache_paths: dict[str, Path],
    vocabulary: list[str],
    window_length: int,
    window_step: int,
    stats: dict[str, Any],
    graph_schema_version: str = graph_schema.GRAPH_SCHEMA_VERSION,
) -> None:
    metadata = {
        "ipn_label_mapping": IPN_TO_VOCAB,
        "vocabulary": vocabulary,
        "window_length": window_length,
        "window_step": window_step,
        "graph_schema_version": graph_schema_version,
        "num_nodes": graph_schema.NUM_NODES,
        "num_coords": graph_schema.NUM_COORDS,
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
        required=False,
        default=None,
        help=(
            "Output path for the generated sequence dataset NPZ.  "
            "Required unless --chunk-dir is used without --output-npz."
        ),
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
            "source label (currently D0X) to limit class imbalance.  "
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
        "--unknown-subsample-ratio",
        type=float,
        default=1.0,
        metavar="R",
        help=(
            "Fraction of unknown-class sequences to keep in the final dataset "
            "(0.0\u20131.0).  Applied after all other filtering.  "
            "Default: 1.0 (keep all)."
        ),
    )
    p.add_argument(
        "--window-length",
        type=int,
        default=32,
        help="Sliding window length in frames (default: 32).",
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
        "--model-path",
        default=None,
        help=(
            "Path to the holistic_landmarker.task model bundle.  "
            "If omitted the model is downloaded to ~/.cache/maestro/ on first use."
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for unknown-segment subsampling (default: 42).",
    )
    p.add_argument(
        "--format",
        default="graph",
        choices=["graph", "flat"],
        help=(
            "Output format: 'graph' (default) produces graph-schema-v1 NPZ with "
            "sequences shape (N,T,67,3); 'flat' produces feature-schema-v5 NPZ with "
            "sequences shape (N,T,147) compatible with TwoStreamGestureLSTM and "
            "GestureDataModule."
        ),
    )
    p.add_argument(
        "--graph-schema-version",
        default=graph_schema.GRAPH_SCHEMA_VERSION,
        help="Graph schema version tag embedded in the NPZ metadata (graph format only).",
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
        "--binary-labels",
        action="store_true",
        default=False,
        help=(
            "Build a binary detector dataset (Phase 3 of the IPN-hand pipeline).  "
            "Remaps all IPN labels to two classes: 'no_gesture' (D0X) and 'gesture' "
            "(all other IPN codes).  Mirrors ipnbinary.json from the IPN-hand paper.  "
            "When set, the default --vocab-config switches to "
            "configs/gestures/vocabulary_ipn_binary.yaml and the default "
            "--window-length switches to 8 (matching the paper's 8-frame detector "
            "window).  Unknown-segment subsampling is disabled in binary mode since "
            "both classes are needed at full coverage."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "If set, save one NPZ per gesture class into this directory "
            "(e.g. data/processed/ipn_chunks/).  "
            "Can be combined with --output-npz to also write the full assembled NPZ."
        ),
    )
    p.add_argument(
        "--assemble-from-chunks",
        type=Path,
        default=None,
        metavar="CHUNK_DIR",
        help=(
            "Fast assembly mode: skip landmark extraction and dataset assembly entirely. "
            "Instead, load pre-built per-class chunk NPZs from CHUNK_DIR and write the "
            "assembled result to --output-npz."
        ),
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.format == "graph" and (
        GraphSequenceDatasetBuilder is None or LabeledGraphFrame is None
    ):
        parser.error(
            "--format graph is unavailable because legacy graph classes are missing. "
            "Use '--format flat'."
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # -----------------------------------------------------------------------
    # Binary-mode defaults: override vocab + window-length when --binary-labels
    # is set but the user has not explicitly supplied those args.
    # -----------------------------------------------------------------------
    active_label_mapping: dict[str, str] | None = None
    if args.binary_labels:
        active_label_mapping = IPN_TO_VOCAB_BINARY
        # Apply binary-mode defaults only when the user has not overridden them.
        # Use parser defaults directly to avoid triggering required-argument
        # validation from parse_args([]).
        default_vocab_config = parser.get_default("vocab_config")
        default_window_length = parser.get_default("window_length")
        if args.vocab_config == default_vocab_config:
            args.vocab_config = "configs/gestures/vocabulary_ipn_binary.yaml"
            logger.info("Binary mode: using default vocab '%s'", args.vocab_config)
        if args.window_length == default_window_length:
            args.window_length = 8
            logger.info("Binary mode: using default window-length 8 (paper detector window)")

    # ── Vocabulary (needed by all modes including fast assembly) ─────────────
    vocab_config = Path(args.vocab_config)
    if not vocab_config.is_file():
        parser.error(f"Vocabulary config not found: {vocab_config}")
    vocabulary = _load_vocabulary(vocab_config)
    logger.info("Vocabulary (%d classes): %s", len(vocabulary), vocabulary)

    # ── Fast path: assemble from pre-built per-class chunk files ─────────────
    if args.assemble_from_chunks:
        if not args.output_npz:
            parser.error("--assemble-from-chunks requires --output-npz to be set.")
        _assemble_ipn_from_chunks(
            chunk_dir=args.assemble_from_chunks.resolve(),
            vocabulary=vocabulary,
            output_path=Path(args.output_npz),
            seed=args.seed,
        )
        return

    # ── Full build: paths needed for landmark extraction ─────────────────────
    ipn_root = Path(args.ipn_root)
    video_dir = ipn_root / "videos"
    annot_path = ipn_root / "annotations" / args.annotation_file
    cache_dir = ipn_root / "cache"
    output_npz = Path(args.output_npz) if args.output_npz else None

    # Validate inputs.
    if not video_dir.is_dir():
        parser.error(f"Video directory not found: {video_dir}")
    if not annot_path.is_file():
        parser.error(f"Annotation file not found: {annot_path}")
    if output_npz is None and not args.chunk_dir:
        parser.error("--output-npz is required unless --chunk-dir is set.")

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    annotations = parse_annotations(annot_path)
    logger.info("Loaded %d annotation rows from %s.", len(annotations), annot_path.name)

    # Collect the unique video stems that appear in (mapped) annotations.
    effective_mapping_for_filter = active_label_mapping if active_label_mapping else IPN_TO_VOCAB
    needed_videos: list[str] = sorted(
        {row.video for row in annotations if row.ipn_label in effective_mapping_for_filter}
    )
    logger.info("%d unique videos needed for mapped labels.", len(needed_videos))

    # -----------------------------------------------------------------------
    # Phase 1 — extract / load per-video landmark caches.
    # -----------------------------------------------------------------------
    # Resolve holistic model path (download if needed).
    # The legacy mp.solutions.holistic.Holistic API used by _extract_video_worker
    # bundles its own model weights inside the mediapipe package and does not
    # require the separate HolisticLandmarker .task file.  We keep the
    # model_path argument for API compatibility but no longer download anything.
    model_path = Path(args.model_path) if args.model_path else _DEFAULT_MODEL_CACHE

    cache_paths = extract_all_landmarks(
        video_dir=video_dir,
        video_stems=needed_videos,
        cache_dir=cache_dir,
        num_workers=num_workers,
        model_path=str(model_path),  # kept for arg-tuple compatibility, unused by worker
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        no_cache=args.no_cache,
    )
    if not cache_paths:
        logger.error(
            "No landmark caches are available.  "
            "Ensure that either (a) .avi video files exist in '%s' or "
            "(b) cached NPZ files exist in '%s'.  "
            "Re-run with --log-level DEBUG for per-video details.",
            video_dir,
            cache_dir,
        )
        sys.exit(1)

    # ── Per-class chunk build (optional, runs before full assembly) ───────────
    if args.chunk_dir:
        _save_ipn_chunks_streaming(
            annotations=annotations,
            cache_paths=cache_paths,
            vocabulary=vocabulary,
            chunk_dir=args.chunk_dir.resolve(),
            window_length=args.window_length,
            window_step=args.window_step,
            min_valid_ratio=args.min_valid_ratio,
            label_strategy=args.label_strategy,
            max_unknown_segments_per_source=args.max_unknown_segments_per_source,
            max_unknown_frames_per_segment=args.max_unknown_frames_per_segment,
            seed=args.seed,
        )
        if output_npz is None:
            return  # chunks-only mode; no full NPZ requested

    # -----------------------------------------------------------------------
    # Phase 2 — assemble sliding-window dataset.
    # -----------------------------------------------------------------------
    assemble_kwargs = dict(
        annotations=annotations,
        cache_paths=cache_paths,
        vocabulary=vocabulary,
        window_length=args.window_length,
        window_step=args.window_step,
        min_valid_ratio=args.min_valid_ratio,
        label_strategy=args.label_strategy,
        max_unknown_segments_per_source=args.max_unknown_segments_per_source,
        max_unknown_frames_per_segment=args.max_unknown_frames_per_segment,
        seed=args.seed,
        label_mapping=active_label_mapping,
    )

    if args.format == "flat":
        mode_desc = (
            "binary detector (2 classes)" if args.binary_labels else "flat feature-schema-v5"
        )
        logger.info("Building %s dataset (right-hand, 147 dims)…", mode_desc)
        builder: SequenceDatasetBuilder | GraphSequenceDatasetBuilder = assemble_flat_dataset(
            **assemble_kwargs
        )
    else:
        logger.info("Building graph-schema-v1 dataset (holistic, 67 nodes × 3 coords)…")
        builder = assemble_dataset(**assemble_kwargs)

    if not builder.samples:
        logger.error(
            "No sequence samples were generated.  Check your annotations, "
            "vocabulary, and window settings."
        )
        sys.exit(1)

    # Post-build unknown-class subsampling (sequence level).
    if args.unknown_subsample_ratio < 1.0:
        rng_sub = random.Random(args.seed)
        all_samples = builder.samples
        unknown_samples = [s for s in all_samples if s.label == "unknown"]
        keep_n = max(1, int(len(unknown_samples) * args.unknown_subsample_ratio))
        rng_sub.shuffle(unknown_samples)
        kept_ids = {id(s) for s in unknown_samples[:keep_n]}
        builder._samples = [
            s for s in builder._samples if s.label != "unknown" or id(s) in kept_ids
        ]
        logger.info(
            "Unknown subsample (ratio=%.2f): %d \u2192 %d sequences",
            args.unknown_subsample_ratio,
            len(unknown_samples),
            len([s for s in builder._samples if s.label == "unknown"]),
        )

    # -----------------------------------------------------------------------
    # Save NPZ and metadata.
    # -----------------------------------------------------------------------
    assert output_npz is not None  # guaranteed: chunks-only path returned above
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "flat":
        assert isinstance(builder, SequenceDatasetBuilder)
        builder.save_npz(output_npz, feature_schema_version="feature-schema-v5")
    else:
        assert GraphSequenceDatasetBuilder is not None
        assert isinstance(builder, GraphSequenceDatasetBuilder)
        builder.save_npz(output_npz, graph_schema_version=args.graph_schema_version)
    logger.info("Dataset saved → %s  (%d samples)", output_npz, len(builder.samples))

    stats = _compute_class_stats(builder)
    logger.info("Class distribution: %s", stats["class_distribution"])

    metadata_path = Path(
        args.metadata_json if args.metadata_json else str(output_npz) + ".metadata.json"
    )
    if args.format == "graph":
        _save_metadata(
            metadata_path=metadata_path,
            annotations_used=len(annotations),
            cache_paths=cache_paths,
            vocabulary=vocabulary,
            window_length=args.window_length,
            window_step=args.window_step,
            stats=stats,
            graph_schema_version=args.graph_schema_version,
        )


if __name__ == "__main__":
    main()
