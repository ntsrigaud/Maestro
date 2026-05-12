"""Build a sliding-window LSTM training dataset from the 20BN-Jester-v1 dataset.

The script follows the same two-phase architecture as ``build_ipn_dataset.py``:

  Phase 1 — Landmark Extraction (parallelised, with per-video caching)
    Each video in the Jester dataset is a **folder of JPEG frames** rather than
    a single video file.  MediaPipe Holistic processes frames in temporal order
    (``static_image_mode=False`` preserves the temporal tracker) and results
    are saved to ``<jester-root>/cache/<video_id>.npz`` for reuse.

  Phase 2 — Dataset Assembly
    Each Jester video folder represents a **complete single gesture** (unlike
    IPN, which has multiple labelled segments per video).  Frames are fed
    sequentially to ``SequenceDatasetBuilder`` using the ``majority`` label
    strategy (the entire recording belongs to one class) with the existing
    feature transformer and sliding-window logic.  The final NPZ is fully
    compatible with ``GestureDataModule`` and ``EnhancedTwoStreamLSTM``.

Usage
-----
    python scripts/build_jester_dataset.py \\
        --jester-root data/20bn-jester-v1 \\
        --output-npz data/processed/jester_pretrain_dataset.npz \\
        --num-workers 4

    # Force re-extraction (ignore existing caches):
    python scripts/build_jester_dataset.py \\
        --jester-root data/20bn-jester-v1 \\
        --output-npz data/processed/jester_pretrain_dataset.npz \\
        --no-cache

    # Build only from the validation split (faster smoke-test):
    python scripts/build_jester_dataset.py \\
        --jester-root data/20bn-jester-v1 \\
        --output-npz data/processed/jester_val_only.npz \\
        --skip-train

Compatibility
-------------
The output NPZ is fully compatible with ``GestureDataModule`` and the existing
training pipeline in ``train_model.py`` / training notebooks.  The vocabulary
is embedded in the NPZ metadata (``metadata_json`` key) and does not need to be
passed separately.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
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

from maestro.pipeline import graph_schema  # noqa: E402
from maestro.training.dataset_builder import (  # noqa: E402
    LabeledLandmarkFrame,
    SequenceDatasetBuilder,
)

logger = logging.getLogger(__name__)


def _ensure_protobuf_compat() -> None:
    """Patch protobuf API changes used by MediaPipe legacy solutions.

    Some environments ship protobuf versions where
    ``symbol_database.Default().GetPrototype`` no longer exists.  Legacy
    MediaPipe solutions still call it internally; we provide a small shim to
    keep dataset builders working across protobuf versions.
    """
    try:
        from google.protobuf import message_factory, symbol_database
    except Exception:
        return

    if hasattr(symbol_database.Default(), "GetPrototype"):
        return

    def _get_prototype(self: Any, descriptor: Any) -> Any:
        get_message_class = getattr(message_factory, "GetMessageClass", None)
        if callable(get_message_class):
            return get_message_class(descriptor)

        # Fallback for older protobuf APIs.
        factory = message_factory.MessageFactory()
        return factory.GetPrototype(descriptor)

    symbol_database.SymbolDatabase.GetPrototype = _get_prototype


# ---------------------------------------------------------------------------
# Jester label → vocabulary key mapping
# ---------------------------------------------------------------------------
# The 27 Jester class strings are mapped to snake_case vocabulary keys.
# The mapping order defines the default label-id assignment (index in the list).

JESTER_LABELS: list[str] = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]

JESTER_TO_VOCAB: dict[str, str] = {
    "Swiping Left": "swiping_left",
    "Swiping Right": "swiping_right",
    "Swiping Down": "swiping_down",
    "Swiping Up": "swiping_up",
    "Pushing Hand Away": "pushing_hand_away",
    "Pulling Hand In": "pulling_hand_in",
    "Sliding Two Fingers Left": "sliding_two_fingers_left",
    "Sliding Two Fingers Right": "sliding_two_fingers_right",
    "Sliding Two Fingers Down": "sliding_two_fingers_down",
    "Sliding Two Fingers Up": "sliding_two_fingers_up",
    "Pushing Two Fingers Away": "pushing_two_fingers_away",
    "Pulling Two Fingers In": "pulling_two_fingers_in",
    "Rolling Hand Forward": "rolling_hand_forward",
    "Rolling Hand Backward": "rolling_hand_backward",
    "Turning Hand Clockwise": "turning_hand_clockwise",
    "Turning Hand Counterclockwise": "turning_hand_counterclockwise",
    "Zooming In With Full Hand": "zooming_in_full_hand",
    "Zooming Out With Full Hand": "zooming_out_full_hand",
    "Zooming In With Two Fingers": "zooming_in_two_fingers",
    "Zooming Out With Two Fingers": "zooming_out_two_fingers",
    "Thumb Up": "thumb_up",
    "Thumb Down": "thumb_down",
    "Shaking Hand": "shaking_hand",
    "Stop Sign": "stop_sign",
    "Drumming Fingers": "drumming_fingers",
    "No gesture": "no_gesture",
    "Doing other things": "doing_other_things",
}

# Canonical 27-class vocabulary (label-id order matches JESTER_LABELS).
DEFAULT_VOCABULARY: list[str] = [JESTER_TO_VOCAB[lbl] for lbl in JESTER_LABELS]

# Jester videos do not embed FPS metadata.  The dataset was captured at
# approximately 12 fps; this value is used only for timestamp_ms computation.
JESTER_FPS: float = 12.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JesterAnnotation:
    """A single Jester annotation record."""

    video_id: str  # e.g. "34870"
    label: str  # original Jester label string, e.g. "Swiping Left"
    vocab_label: str  # vocabulary key, e.g. "swiping_left"


@dataclass
class VideoLandmarks:
    """Per-frame holistic landmark results for a single Jester video."""

    video_id: str
    fps: float
    frame_ids: np.ndarray  # (N,) int32 — 0-indexed frame order
    keypoints: np.ndarray  # (N, 67, 3) float32 — pre-normalisation
    valid_mask: np.ndarray  # (N, 67) bool — per-node detection validity


# ---------------------------------------------------------------------------
# Landmark parsing — identical to build_ipn_dataset.py
# ---------------------------------------------------------------------------


def _is_nan(lm: object) -> bool:
    import math

    return (
        math.isnan(getattr(lm, "x", 0.0))
        or math.isnan(getattr(lm, "y", 0.0))
        or math.isnan(getattr(lm, "z", 0.0))
    )


def _parse_holistic_legacy(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Parse a MediaPipe Holistic result into (67, 3) keypoints + valid mask.

    Populates the full 67-node graph-schema-v1 layout:
      nodes  0–20 : right hand
      nodes 21–41 : left hand
      nodes 42–52 : 11 pose landmarks
      nodes 53–66 : 14 face mesh contour points
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


# ---------------------------------------------------------------------------
# Phase 1 — per-video landmark extraction (worker)
# ---------------------------------------------------------------------------


def _extract_jester_video_worker(
    args: tuple[str, str, float, float, float],
) -> tuple[str, str]:
    """Extract holistic landmarks from a Jester JPEG-frame folder.

    Unlike IPN (single .avi file), Jester stores each video as a folder of
    JPEG frames named ``00001.jpg``, ``00002.jpg``, etc.  This worker reads
    them in sorted order and runs MediaPipe Holistic with
    ``static_image_mode=False`` so the temporal tracker is preserved.

    Parameters
    ----------
    args:
        ``(video_folder_path, cache_path, min_detection_confidence,
        min_tracking_confidence, fps)``

    Returns
    -------
    ``(video_id, cache_path_str)``
        On failure ``cache_path_str`` is an empty string.
    """
    _ensure_protobuf_compat()

    import mediapipe as mp

    video_folder_str, cache_path_str, min_det_conf, min_track_conf, fps = args
    video_folder = Path(video_folder_str)
    cache_path = Path(cache_path_str)
    video_id = video_folder.name

    if cache_path.exists():
        return video_id, cache_path_str

    jpg_files = sorted(video_folder.glob("*.jpg"))
    if not jpg_files:
        logger.error("No JPEG frames in: %s", video_folder)
        return video_id, ""

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
        for frame_idx, jpg_path in enumerate(jpg_files):
            bgr = cv2.imread(str(jpg_path))
            if bgr is None:
                logger.warning("Cannot read frame: %s", jpg_path)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            kp, vm = _parse_holistic_legacy(result)
            frame_ids.append(frame_idx)
            keypoints_list.append(kp)
            valid_mask_list.append(vm)

    if not frame_ids:
        logger.error("No frames could be processed: %s", video_folder)
        return video_id, ""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fps=np.float32(fps),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        keypoints=np.asarray(keypoints_list, dtype=np.float32),
        valid_mask=np.asarray(valid_mask_list, dtype=bool),
    )
    logger.debug("Cached %d frames → %s", len(frame_ids), cache_path.name)
    return video_id, cache_path_str


def _load_cache(cache_path: Path) -> VideoLandmarks:
    """Load a cached per-video NPZ (same format as build_ipn_dataset.py)."""
    with np.load(cache_path, allow_pickle=True) as data:
        return VideoLandmarks(
            video_id=cache_path.stem,
            fps=float(data["fps"]),
            frame_ids=data["frame_ids"],
            keypoints=data["keypoints"],  # (N, 67, 3)
            valid_mask=data["valid_mask"],  # (N, 67)
        )


def extract_all_landmarks(
    video_root: Path,
    video_ids: list[str],
    cache_dir: Path,
    *,
    num_workers: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    fps: float,
    no_cache: bool,
) -> dict[str, Path]:
    """Extract or load cached holistic landmarks for every requested video.

    Returns a dict mapping ``video_id`` → cache NPZ ``Path``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, float, float, float]] = []
    result_paths: dict[str, Path] = {}

    for vid_id in video_ids:
        cache_path = cache_dir / f"{vid_id}.npz"

        if not no_cache and cache_path.exists():
            result_paths[vid_id] = cache_path
            continue

        video_folder = video_root / vid_id
        if not video_folder.is_dir():
            logger.warning("Video folder not found, skipping: %s", video_folder)
            continue

        if no_cache and cache_path.exists():
            cache_path.unlink()

        tasks.append(
            (
                str(video_folder),
                str(cache_path),
                min_detection_confidence,
                min_tracking_confidence,
                fps,
            )
        )

    if result_paths:
        logger.info(
            "%d video(s) resolved from cache; %d queued for extraction.",
            len(result_paths),
            len(tasks),
        )

    if not tasks and not result_paths:
        logger.warning("No videos to process and no existing caches found.")
        return {}

    logger.info("Extracting %d video(s) with %d worker(s)…", len(tasks), num_workers)
    t0 = time.perf_counter()

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Extracting landmarks", unit="video"):
            vid_id, cpath = _extract_jester_video_worker(task)
            if cpath:
                result_paths[vid_id] = Path(cpath)
    else:
        chunksize = max(1, min(32, len(tasks) // max(1, num_workers * 8)))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            iterator = executor.map(_extract_jester_video_worker, tasks, chunksize=chunksize)
            for vid_id, cpath in tqdm(
                iterator, total=len(tasks), desc="Extracting landmarks", unit="video"
            ):
                if cpath:
                    result_paths[vid_id] = Path(cpath)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Extraction complete in %.1fs  (%d / %d videos cached).",
        elapsed,
        len(result_paths),
        len(video_ids),
    )
    return result_paths


# ---------------------------------------------------------------------------
# Phase 2 — dataset assembly
# ---------------------------------------------------------------------------


def parse_annotations(
    annot_path: Path,
    vocabulary_set: set[str],
) -> list[JesterAnnotation]:
    """Parse a Jester train or validation CSV into ``JesterAnnotation`` records.

    The CSV has no header and uses ``;`` as separator::

        {video_id};{jester_label}

    Labels not present in *vocabulary_set* (after ``JESTER_TO_VOCAB`` mapping)
    are silently skipped.  The Jester test CSV contains only video IDs (no
    labels) and is not supported here.
    """
    rows: list[JesterAnnotation] = []
    with annot_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";", 1)
            if len(parts) != 2:
                continue  # test CSV (IDs only) — skip silently

            video_id, jester_label = parts[0].strip(), parts[1].strip()
            vocab_label = JESTER_TO_VOCAB.get(jester_label)
            if vocab_label is None:
                logger.warning("Unknown Jester label: %r (video %s)", jester_label, video_id)
                continue
            if vocab_label not in vocabulary_set:
                continue

            rows.append(
                JesterAnnotation(
                    video_id=video_id,
                    label=jester_label,
                    vocab_label=vocab_label,
                )
            )
    return rows


def assemble_dataset(
    annotations: list[JesterAnnotation],
    cache_paths: dict[str, Path],
    vocabulary: list[str],
    *,
    window_length: int,
    window_step: int,
    min_valid_ratio: float,
    label_strategy: str,
    max_samples_per_video: int,
    seed: int,
) -> SequenceDatasetBuilder:
    """Feed Jester landmark frames into a ``SequenceDatasetBuilder``.

    Each Jester video is a complete single-gesture recording; there are no
    intra-video segment boundaries.  All frames are fed sequentially under a
    single ``recording_id``.  The builder resets its internal state when the
    recording ID changes (i.e. at every new video), ensuring sliding windows
    do not span across recordings.

    Parameters
    ----------
    max_samples_per_video:
        Hard cap on the number of sliding-window samples emitted per video.
        Once reached, frame feeding stops early for that recording.
        Set to ``0`` to disable.  Strongly recommended for the ``No gesture``
        and ``Doing other things`` classes whose recordings tend to be longer
        than typical gesture classes.
    """
    builder = SequenceDatasetBuilder(
        vocabulary=vocabulary,
        window_length=window_length,
        window_step=window_step,
        min_valid_ratio=min_valid_ratio,
        label_strategy=label_strategy,
    )

    total_windows = 0
    missing_cache = 0

    for ann in tqdm(annotations, desc="Assembling dataset", unit="video"):
        cache_path = cache_paths.get(ann.video_id)
        if cache_path is None or not cache_path.exists():
            missing_cache += 1
            continue

        vl = _load_cache(cache_path)
        n_frames = len(vl.frame_ids)

        emitted_this_video = 0
        for frame_idx in range(n_frames):
            # Respect per-video cap — stop adding frames once saturated.
            if max_samples_per_video > 0 and emitted_this_video >= max_samples_per_video:
                break

            timestamp_ms = int(frame_idx * 1000.0 / vl.fps)
            rh_kps = vl.keypoints[frame_idx, :21]  # (21, 3)
            rh_valid = bool(vl.valid_mask[frame_idx, :21].any())

            lf = LabeledLandmarkFrame(
                recording_id=ann.video_id,
                timestamp_ms=timestamp_ms,
                label=ann.vocab_label,
                landmarks=rh_kps.tolist(),
                valid=rh_valid,
                handedness="Right",
            )
            emitted = builder.add_frame(lf)
            emitted_this_video += len(emitted)

        total_windows += emitted_this_video

    if missing_cache > 0:
        logger.warning("%d video(s) had no cache and were skipped.", missing_cache)

    logger.info(
        "Jester assembly complete: %d sliding-window samples from %d videos.",
        total_windows,
        len(annotations) - missing_cache,
    )
    return builder


def _compute_class_stats(builder: SequenceDatasetBuilder) -> dict[str, Any]:
    label_counts = Counter(s.label for s in builder.samples)
    return {
        "total_samples": len(builder.samples),
        "class_distribution": dict(label_counts),
        "num_classes": len(label_counts),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a Jester sliding-window dataset for EnhancedTwoStreamLSTM pretraining."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jester-root",
        type=Path,
        default=Path("data/20bn-jester-v1"),
        help=(
            "Root of the 20BN-Jester-v1 dataset. "
            "Must contain annotations/ and videos/20bn-jester-v1/ subdirectories."
        ),
    )
    p.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/processed/jester_pretrain_dataset.npz"),
        help="Output path for the NPZ dataset.",
    )
    p.add_argument(
        "--vocab-config",
        type=Path,
        default=Path("configs/gestures/vocabulary_jester.yaml"),
        help="YAML vocabulary config defining the Jester classes to include.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Per-video landmark cache directory. Defaults to <jester-root>/cache/.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel worker processes for landmark extraction.",
    )
    p.add_argument(
        "--window-length",
        type=int,
        default=32,
        help="Sliding-window length in frames.",
    )
    p.add_argument(
        "--window-step",
        type=int,
        default=1,
        help="Sliding-window stride.",
    )
    p.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.6,
        help="Minimum fraction of valid frames required to emit a window.",
    )
    p.add_argument(
        "--label-strategy",
        choices=["last", "majority"],
        default="majority",
        help=(
            "Label assignment strategy. 'majority' is recommended for Jester: "
            "each recording is a single complete gesture."
        ),
    )
    p.add_argument(
        "--max-samples-per-video",
        type=int,
        default=20,
        help=(
            "Maximum sliding-window samples per video. "
            "Limits long 'Doing other things' recordings. Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=JESTER_FPS,
        help="Frame rate for timestamp_ms calculation (Jester ≈ 12 fps).",
    )
    p.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    p.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing per-video caches and re-extract all landmarks.",
    )
    p.add_argument(
        "--exclude-classes",
        type=str,
        default="",
        metavar="CLASS1,CLASS2,...",
        help=(
            "Comma-separated list of vocabulary class names to exclude from the output. "
            "Use this to prevent data leakage when the Jester dataset is used as a "
            "pretraining source before fine-tuning on a hybrid vocabulary that shares "
            "some of the same Jester classes (e.g. swiping_left, swiping_right, "
            "stop_sign, no_gesture, zooming_in_full_hand, zooming_out_full_hand, "
            "swiping_down, swiping_up)."
        ),
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training split (use validation only).",
    )
    p.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip the validation split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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

    jester_root = args.jester_root.resolve()
    annot_dir = jester_root / "annotations"
    video_root = jester_root / "videos" / "20bn-jester-v1"
    cache_dir = args.cache_dir or (jester_root / "cache")

    logger.info("Jester root  : %s", jester_root)
    logger.info("Video dir    : %s", video_root)
    logger.info("Cache dir    : %s", cache_dir)
    logger.info("Output NPZ   : %s", args.output_npz)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    if args.vocab_config.exists():
        with args.vocab_config.open(encoding="utf-8") as vf:
            _vcfg = yaml.safe_load(vf) or {}
        vocabulary = [str(g) for g in (_vcfg.get("gestures") or [])]
    else:
        vocabulary = []

    if not vocabulary:
        vocabulary = list(DEFAULT_VOCABULARY)
        logger.warning(
            "Vocabulary config not found or empty — using default 27-class Jester vocabulary."
        )

    vocabulary_set = set(vocabulary)
    logger.info("Vocabulary   : %d classes", len(vocabulary))

    # ── Class exclusion ───────────────────────────────────────────────────────
    excluded_classes: set[str] = set()
    if args.exclude_classes.strip():
        excluded_classes = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
        invalid = excluded_classes - vocabulary_set
        if invalid:
            logger.warning(
                "--exclude-classes contains labels not in vocabulary (ignored): %s", invalid
            )
        excluded_classes &= vocabulary_set
        logger.info(
            "Excluding %d class(es) from output: %s",
            len(excluded_classes),
            sorted(excluded_classes),
        )

    # ── Annotations ───────────────────────────────────────────────────────────
    all_annotations: list[JesterAnnotation] = []

    if not args.skip_train:
        train_csv = annot_dir / "jester-v1-train.csv"
        if train_csv.exists():
            train_anns = parse_annotations(train_csv, vocabulary_set)
            logger.info("Train split  : %d videos loaded", len(train_anns))
            all_annotations.extend(train_anns)
        else:
            logger.warning("Train CSV not found: %s", train_csv)

    if not args.skip_val:
        val_csv = annot_dir / "jester-v1-validation.csv"
        if val_csv.exists():
            val_anns = parse_annotations(val_csv, vocabulary_set)
            logger.info("Val split    : %d videos loaded", len(val_anns))
            all_annotations.extend(val_anns)
        else:
            logger.warning("Validation CSV not found: %s", val_csv)

    if not all_annotations:
        logger.error("No annotations loaded. Aborting.")
        sys.exit(1)

    # Apply class exclusion filter.
    if excluded_classes:
        before = len(all_annotations)
        all_annotations = [a for a in all_annotations if a.vocab_label not in excluded_classes]
        logger.info(
            "After exclusion: %d videos retained  (%d excluded)",
            len(all_annotations),
            before - len(all_annotations),
        )
        # Rebuild vocabulary without excluded classes so the embedded metadata
        # and label_id assignments reflect only the classes actually present.
        vocabulary = [v for v in vocabulary if v not in excluded_classes]
        vocabulary_set = set(vocabulary)

    logger.info(
        "Total        : %d videos across %d classes",
        len(all_annotations),
        len({a.vocab_label for a in all_annotations}),
    )

    # ── Phase 1: Landmark extraction ──────────────────────────────────────────
    video_ids = [a.video_id for a in all_annotations]
    cache_paths = extract_all_landmarks(
        video_root=video_root,
        video_ids=video_ids,
        cache_dir=cache_dir,
        num_workers=args.num_workers,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        fps=args.fps,
        no_cache=args.no_cache,
    )

    # ── Phase 2: Dataset assembly ──────────────────────────────────────────────
    logger.info(
        "Assembling dataset  (window=%d  step=%d  strategy=%s  max_per_video=%d)…",
        args.window_length,
        args.window_step,
        args.label_strategy,
        args.max_samples_per_video,
    )
    builder = assemble_dataset(
        annotations=all_annotations,
        cache_paths=cache_paths,
        vocabulary=vocabulary,
        window_length=args.window_length,
        window_step=args.window_step,
        min_valid_ratio=args.min_valid_ratio,
        label_strategy=args.label_strategy,
        max_samples_per_video=args.max_samples_per_video,
        seed=args.seed,
    )

    if not builder.samples:
        logger.error("No samples generated — check cache coverage and vocabulary.")
        sys.exit(1)

    stats = _compute_class_stats(builder)
    logger.info("Total samples: %d", stats["total_samples"])
    logger.info("Class distribution:")
    for cls, cnt in sorted(stats["class_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * min(40, cnt // max(1, stats["total_samples"] // 800))
        logger.info("  %-40s %6d  %s", cls, cnt, bar)

    # ── Save NPZ ───────────────────────────────────────────────────────────────
    logger.info("Saving NPZ → %s …", args.output_npz)
    builder.save_npz(args.output_npz, feature_schema_version="feature-schema-v5")
    logger.info("Dataset saved: %s  (%d samples)", args.output_npz, stats["total_samples"])

    # ── Save metadata sidecar ─────────────────────────────────────────────────
    meta_path = args.output_npz.with_suffix("").with_suffix("").parent / (
        args.output_npz.stem + ".metadata.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "jester_label_mapping": JESTER_TO_VOCAB,
                "vocabulary": vocabulary,
                "window_length": args.window_length,
                "window_step": args.window_step,
                "fps": args.fps,
                "min_valid_ratio": args.min_valid_ratio,
                "label_strategy": args.label_strategy,
                "max_samples_per_video": args.max_samples_per_video,
                "videos_processed": len(cache_paths),
                "annotations_loaded": len(all_annotations),
                "dataset_stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Metadata written to %s", meta_path)


if __name__ == "__main__":
    main()
