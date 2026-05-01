"""STGCN-LSTM parity and end-to-end integration tests.

Verifies that:
1. A freshly constructed STGCN_LSTM model produces deterministic output for a
   fixed random input.
2. The export → checkpoint-load round-trip preserves model weights and metadata.
3. GraphGestureDataModule loads a synthetic NPZ correctly.
4. The Runtime hot-loop can run N frames using the STGCN pipeline (no camera
   required — all I/O is mocked).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM
from maestro.pipeline.graph_schema import (
    NUM_COORDS,
    NUM_NODES,
    WINDOW_LENGTH,
    WINDOW_STEP,
    build_adjacency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_CLASSES = 8
_BATCH = 2


def _make_model(num_classes: int = _NUM_CLASSES) -> STGCN_LSTM:
    return STGCN_LSTM(num_classes=num_classes)


def _random_batch(
    batch: int = _BATCH,
    t: int = WINDOW_LENGTH,
    n: int = NUM_NODES,
    c: int = NUM_COORDS,
    seed: int = 0,
) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(batch, t, n, c, generator=rng)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


class TestSTGCNDeterminism:
    def test_same_input_same_output(self) -> None:
        model = _make_model()
        model.eval()
        x = _random_batch()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Model output is not deterministic"

    def test_output_shape(self) -> None:
        model = _make_model()
        model.eval()
        x = _random_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (_BATCH, _NUM_CLASSES)

    def test_output_is_logits_not_probs(self) -> None:
        model = _make_model()
        model.eval()
        x = _random_batch()
        with torch.no_grad():
            out = model(x)
        # At least some logits should be outside [0, 1].
        assert not ((out >= 0) & (out <= 1)).all(), "Output looks like probs, expected raw logits"


# ---------------------------------------------------------------------------
# 2. Export → load round-trip
# ---------------------------------------------------------------------------


class TestExportLoadRoundtrip:
    def _make_result(
        self,
        model: STGCN_LSTM,
        tmp_path: Path,
    ) -> object:
        """Build a minimal TrainingResult-like namespace."""
        from maestro.training.train_lstm import EvaluationReport, TrainingResult

        dummy_report = EvaluationReport(
            loss=0.0,
            accuracy=0.0,
            macro_f1=0.0,
            per_class_precision={},
            per_class_recall={},
            confusion_matrix=[],
            calibration_ece=0.0,
            mean_confidence=0.0,
        )
        return TrainingResult(
            model=model,
            model_version="stgcn_test_v1",
            checkpoint_path=tmp_path / "dummy.pt",
            run_dir=tmp_path,
            train_history=[],
            val_report=dummy_report,
            test_report=dummy_report,
            class_labels=[f"cls_{i}" for i in range(_NUM_CLASSES)],
            dataset_metadata={
                "window_length": WINDOW_LENGTH,
                "window_step": WINDOW_STEP,
                "num_nodes": NUM_NODES,
                "num_coords": NUM_COORDS,
                "graph_schema_version": "graph-schema-v1",
            },
        )

    def _make_config(self) -> object:
        """Build a minimal TrainingPipelineConfig-like namespace."""
        return SimpleNamespace(
            model=SimpleNamespace(
                model_name="stgcn_lstm_v1",
                num_nodes=NUM_NODES,
                num_coords=NUM_COORDS,
                stgcn_channels=(64, 128),
                num_stgcn_layers=2,
                lstm_hidden=128,
                lstm_layers=2,
                dropout=0.3,
                num_classes=_NUM_CLASSES,
            ),
            export=SimpleNamespace(
                graph_schema_version="graph-schema-v1",
                feature_schema_version="feature-schema-v5",
            ),
        )

    def test_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact
        from maestro.training.export_checkpoint import export_inference_artifact

        model = _make_model()
        model.eval()

        result = self._make_result(model, tmp_path)
        config = self._make_config()

        artifact_path = export_inference_artifact(
            result=result,
            config=config,
            export_dir=tmp_path,
        )
        assert artifact_path.exists()

        device = torch.device("cpu")
        loaded = load_inference_artifact(artifact_path=artifact_path, device=device)

        assert loaded.graph_schema_version == "graph-schema-v1"
        assert loaded.num_nodes == NUM_NODES
        assert loaded.num_coords == NUM_COORDS
        assert loaded.window_length == WINDOW_LENGTH
        assert loaded.window_step == WINDOW_STEP
        assert loaded.class_labels == [f"cls_{i}" for i in range(_NUM_CLASSES)]

        # Weights must be identical.
        x = _random_batch()
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded.model(x)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)

    def test_roundtrip_artifact_type(self, tmp_path: Path) -> None:
        from maestro.training.export_checkpoint import export_inference_artifact

        model = _make_model()
        result = self._make_result(model, tmp_path)
        config = self._make_config()

        artifact_path = export_inference_artifact(result=result, config=config, export_dir=tmp_path)
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        assert payload["artifact_type"] == "stgcn_lstm_inference"
        assert "graph_schema" in payload
        assert "feature_schema" not in payload


# ---------------------------------------------------------------------------
# 3. GraphGestureDataModule synthetic NPZ
# ---------------------------------------------------------------------------


class TestGraphDataModuleSynthetic:
    def _make_npz(self, tmp_path: Path, num_samples: int = 20) -> Path:
        """Write a minimal graph NPZ matching dataset_builder.save_npz() format."""
        T, N, C = WINDOW_LENGTH, NUM_NODES, NUM_COORDS
        labels = ["swipe_up", "swipe_down"]
        rng = np.random.default_rng(42)

        sequences = rng.random((num_samples, T, N, C)).astype(np.float32)
        label_ids = np.array([i % len(labels) for i in range(num_samples)], dtype=np.int64)
        label_strs = np.array([labels[i % len(labels)] for i in range(num_samples)])
        recording_ids = np.array([f"rec_{i}" for i in range(num_samples)])
        valid_masks = np.ones((num_samples, T, N), dtype=bool)
        adjacency = build_adjacency().astype(np.float32)

        metadata = {
            "graph_schema_version": "graph-schema-v1",
            "num_nodes": N,
            "num_coords": C,
            "window_length": T,
            "window_step": WINDOW_STEP,
            "class_labels": labels,
            "num_samples": num_samples,
        }

        path = tmp_path / "synthetic_graph.npz"
        np.savez_compressed(
            path,
            sequences=sequences,
            label_ids=label_ids,
            labels=label_strs,
            recording_ids=recording_ids,
            valid_masks=valid_masks,
            adjacency=adjacency,
            metadata_json=np.array(json.dumps(metadata)),
        )
        return path

    def test_datamodule_loads_and_splits(self, tmp_path: Path) -> None:
        from maestro.training.datamodule import DataSplitConfig, GraphGestureDataModule

        npz_path = self._make_npz(tmp_path, num_samples=20)
        dm = GraphGestureDataModule(
            dataset_path=npz_path,
            batch_size=4,
            split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
            seed=0,
            num_workers=0,
        )
        dm.setup()

        assert dm.num_classes == 2
        assert dm.num_nodes == NUM_NODES
        assert dm.num_coords == NUM_COORDS
        assert dm.sequence_length == WINDOW_LENGTH

        train_dl = dm.train_dataloader()
        batch = next(iter(train_dl))
        x, y = batch
        assert x.shape[1:] == (WINDOW_LENGTH, NUM_NODES, NUM_COORDS)
        assert y.dtype == torch.int64


# ---------------------------------------------------------------------------
# 4. Runtime hot-loop (STGCN path, no camera)
# ---------------------------------------------------------------------------


class TestSTGCNRuntimeHotLoop:
    def _make_container(
        self, model: STGCN_LSTM, num_nodes: int = NUM_NODES, num_coords: int = NUM_COORDS
    ) -> SimpleNamespace:
        """Build a minimal container that exercises the STGCN branch in Runtime."""
        from maestro.domain.entities import HolisticPacket
        from maestro.pipeline.graph_buffering import GraphSequenceBuffer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_source = MagicMock()
        frame_source.read.return_value = frame

        gaze_detector = MagicMock()
        gaze_detector.update.return_value = True

        # HolisticLandmarkExtractor mock — returns valid keypoints.
        keypoints = np.zeros((num_nodes, num_coords), dtype=np.float32)
        valid_mask = np.ones(num_nodes, dtype=bool)
        holistic_packet = HolisticPacket(keypoints=keypoints, valid_mask=valid_mask, timestamp_ms=0)

        holistic_extractor = MagicMock()
        holistic_extractor.extract.return_value = holistic_packet

        graph_buffer = GraphSequenceBuffer(
            window_length=WINDOW_LENGTH,
            window_step=WINDOW_STEP,
        )

        from maestro.infrastructure.model.checkpoint_loader import InferenceArtifact

        artifact = InferenceArtifact(
            model=model,
            model_version="test",
            class_labels=[f"cls_{i}" for i in range(_NUM_CLASSES)],
            artifact_path=Path("/dev/null"),
            graph_schema_version="graph-schema-v1",
            num_nodes=num_nodes,
            num_coords=num_coords,
            window_length=WINDOW_LENGTH,
            window_step=WINDOW_STEP,
        )

        activation_controller = MagicMock()
        activation_controller.mode = "idle"
        activation_controller.route.return_value = None

        stabilizer = MagicMock()
        pred = MagicMock()
        pred.label = "unknown"
        pred.confidence = 0.1
        pred.accepted = False
        stabilizer.update.return_value = pred

        overlay_frame = MagicMock()
        overlay_frame.frame = frame
        overlay_renderer = MagicMock()
        overlay_renderer.render.return_value = overlay_frame

        # Required by RuntimeDependencies but not used in STGCN path.
        landmark_extractor = MagicMock()
        hand_selector = MagicMock()
        feature_transformer = MagicMock()
        sequence_buffer = MagicMock()

        return SimpleNamespace(
            config=SimpleNamespace(
                app=SimpleNamespace(app_name="test-stgcn", navigation_mode="workspace"),
                inference=SimpleNamespace(gaze=SimpleNamespace(sample_interval_frames=1)),
            ),
            frame_source=frame_source,
            gaze_detector=gaze_detector,
            landmark_extractor=landmark_extractor,
            hand_selector=hand_selector,
            feature_transformer=feature_transformer,
            sequence_buffer=sequence_buffer,
            model_artifact=artifact,
            stabilizer=stabilizer,
            activation_controller=activation_controller,
            dispatcher=MagicMock(),
            overlay_renderer=overlay_renderer,
            benchmark=None,
            show_window=False,
            holistic_extractor=holistic_extractor,
            graph_buffer=graph_buffer,
        )

    def test_hot_loop_runs_n_frames(self) -> None:
        from maestro.app.runtime import Runtime

        model = _make_model()
        model.eval()
        container = self._make_container(model)

        states: list = []
        runtime = Runtime(container, on_state=states.append)

        # Run enough frames to trigger at least one window emission.
        n_frames = WINDOW_LENGTH + 5
        runtime.run(max_frames=n_frames)

        assert len(states) == n_frames, f"Expected {n_frames} states, got {len(states)}"

    def test_hot_loop_emits_window(self) -> None:
        from maestro.app.runtime import Runtime

        model = _make_model()
        model.eval()
        container = self._make_container(model)

        stabilizer_calls: list[int] = []

        original_update = container.stabilizer.update

        def counting_update(pred):  # noqa: E306
            stabilizer_calls.append(1)
            return original_update(pred)

        container.stabilizer.update = counting_update

        # WINDOW_LENGTH frames to fill, then WINDOW_STEP frames to trigger 2nd window.
        n_frames = WINDOW_LENGTH + WINDOW_STEP
        runtime = Runtime(container)
        runtime.run(max_frames=n_frames)

        assert len(stabilizer_calls) >= 2, (
            f"Expected ≥2 window emissions, got {len(stabilizer_calls)}"
        )
