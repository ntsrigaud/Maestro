"""Integration tests for GraphGestureDataModule.

Verifies that GraphGestureDataModule correctly loads, validates and splits a
synthetic graph-NPZ file in the same format produced by dataset_builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from maestro.pipeline.graph_schema import (
    NUM_COORDS,
    NUM_NODES,
    WINDOW_LENGTH,
    WINDOW_STEP,
    build_adjacency,
)


@pytest.mark.training
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
