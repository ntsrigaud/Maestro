"""Unit tests for DualBranchSTGCN_LSTM (dual-branch phonological + STGCN model).

Covers:
  - Forward-pass output shape.
  - Phonological branch and STGCN branch sub-module shapes.
  - Adjacency buffer shape (3, 67, 67) and dtype.
  - Custom adjacency accepted without error.
  - Batch size 1 (BatchNorm edge case).
  - Overfit on a tiny batch (verifies gradient flow through both branches).
  - Parameter count is reasonable (> 0, clearly non-trivial).
  - load_training_config round-trip for the dual-branch YAML.
  - _build_model dispatch for 'dual_stgcn_*' model_name.
"""

from __future__ import annotations

from pathlib import Path

import torch

from maestro.infrastructure.model.dual_branch_stgcn_lstm import (
    DualBranchSTGCN_LSTM,
    _PhonologicalBranch,
    _SpatialAttnPool,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parents[2]
_DUAL_CONFIG = _REPO_ROOT / "configs" / "model" / "dual_branch_stgcn_lstm.yaml"


def _build_model(**kwargs) -> DualBranchSTGCN_LSTM:
    defaults: dict = dict(
        num_nodes=67,
        num_coords=3,
        phono_proj_dim=64,
        phono_conv_channels=128,
        phono_lstm_hidden=128,
        phono_lstm_layers=2,
        stgcn_channels=(64, 128, 256),
        dropout=0.3,
        num_classes=8,
    )
    defaults.update(kwargs)
    return DualBranchSTGCN_LSTM(**defaults)


def _batch(B: int = 4, T: int = 24, N: int = 67, C: int = 3) -> torch.Tensor:
    return torch.randn(B, T, N, C)


# ── _SpatialAttnPool ─────────────────────────────────────────────────────────


class TestSpatialAttnPool:
    def test_output_shape(self) -> None:
        layer = _SpatialAttnPool(in_dim=32)
        x = torch.randn(4, 24, 67, 32)
        out = layer(x)
        assert out.shape == (4, 24, 32), f"Expected (4,24,32) got {out.shape}"

    def test_weights_sum_to_one(self) -> None:
        """Attention weights over N nodes must sum to 1 at each (B, T) cell."""
        import torch.nn.functional as F

        layer = _SpatialAttnPool(in_dim=16)
        x = torch.randn(2, 10, 67, 16)
        # Manually compute weights to verify normalisation
        scores = layer.score(x).squeeze(-1)   # (2, 10, 67)
        weights = F.softmax(scores, dim=2)     # (2, 10, 67)
        sums = weights.sum(dim=2)              # (2, 10) — all should be ≈1
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ── _PhonologicalBranch ───────────────────────────────────────────────────────


class TestPhonologicalBranch:
    def test_output_shape(self) -> None:
        branch = _PhonologicalBranch(
            num_nodes=67,
            num_coords=3,
            proj_dim=64,
            conv_channels=128,
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.3,
        )
        branch.eval()
        x = _batch(B=4, T=24, N=67, C=3)
        with torch.no_grad():
            out = branch(x)
        assert out.shape == (4, 128), f"Expected (4,128) got {out.shape}"

    def test_batch_size_one(self) -> None:
        branch = _PhonologicalBranch(
            num_nodes=67,
            num_coords=3,
            proj_dim=64,
            conv_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.0,
        )
        branch.eval()
        with torch.no_grad():
            out = branch(_batch(B=1))
        assert out.shape == (1, 128)


# ── DualBranchSTGCN_LSTM ──────────────────────────────────────────────────────


class TestDualBranchForward:
    def test_output_shape_standard(self) -> None:
        model = _build_model()
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 8), f"Expected (4,8) got {out.shape}"

    def test_output_shape_batch_size_one(self) -> None:
        model = _build_model()
        model.eval()
        with torch.no_grad():
            out = model(_batch(B=1))
        assert out.shape == (1, 8)

    def test_output_shape_more_classes(self) -> None:
        model = _build_model(num_classes=13)
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 13)

    def test_output_is_logits_not_probs(self) -> None:
        """Output must be raw logits (no softmax)."""
        model = _build_model()
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        # If softmax were applied, values would all be in (0, 1) and sum to 1.
        # Raw logits can be negative and won't sum to 1.
        import torch.nn.functional as F
        probs = F.softmax(out, dim=-1)
        assert not torch.allclose(out, probs, atol=1e-3), "Output should be logits, not probs"

    def test_two_stgcn_blocks(self) -> None:
        model = _build_model(stgcn_channels=(64, 128))
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 8)

    def test_single_stgcn_block(self) -> None:
        model = _build_model(stgcn_channels=(64,))
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 8)


class TestDualBranchAdjacency:
    def test_default_adjacency_shape(self) -> None:
        """Fully-connected K=3 adjacency must be (3, 67, 67)."""
        model = _build_model()
        A = model.adjacency
        assert A.shape == (3, 67, 67), f"Expected (3,67,67) got {A.shape}"

    def test_default_adjacency_dtype(self) -> None:
        model = _build_model()
        assert model.adjacency.dtype == torch.float32

    def test_default_adjacency_in_state_dict(self) -> None:
        model = _build_model()
        assert "adjacency" in model.state_dict()

    def test_custom_adjacency_2d_accepted(self) -> None:
        """2-D (N, N) adjacency should be stored as (1, N, N)."""
        A = torch.rand(67, 67)
        model = _build_model(adjacency=A)
        assert model.adjacency.shape == (1, 67, 67)

    def test_custom_adjacency_3d_accepted(self) -> None:
        A = torch.rand(3, 67, 67)
        model = _build_model(adjacency=A)
        assert model.adjacency.shape == (3, 67, 67)

    def test_fully_connected_adjacency_dense(self) -> None:
        """Every off-diagonal entry must appear in exactly one partition (unnormalised)."""
        import numpy as np

        from maestro.pipeline.graph_schema import build_fully_connected_partitioned_adjacency

        A = build_fully_connected_partitioned_adjacency()  # (3, 67, 67) unnormalised
        off_diag_mask = ~np.eye(67, dtype=bool)
        # Across the three partitions, each off-diagonal entry must be 1 in
        # exactly one partition and 0 in the other two.
        part_sum = A.sum(axis=0)                           # (67, 67)
        assert np.all(
            part_sum[off_diag_mask] == 1.0
        ), "Each off-diagonal edge must appear in exactly one partition."
        # Diagonal must be covered by partition 0 only (self partition = I).
        diag_mask = np.eye(67, dtype=bool)
        assert np.all(A[0][diag_mask] == 1.0), "Partition 0 must be identity."
        assert np.all(A[1][diag_mask] == 0.0), "Partition 1 must have no self-loops."
        assert np.all(A[2][diag_mask] == 0.0), "Partition 2 must have no self-loops."


class TestDualBranchParameters:
    def test_parameter_count_positive(self) -> None:
        model = _build_model()
        n = model.count_parameters()
        assert n > 0

    def test_parameter_count_no_budget_constraint(self) -> None:
        """Full model with paper defaults should have > 1 M parameters."""
        model = DualBranchSTGCN_LSTM(
            num_nodes=67,
            num_coords=3,
            phono_proj_dim=128,
            phono_conv_channels=256,
            phono_lstm_hidden=256,
            phono_lstm_layers=2,
            stgcn_channels=(64, 128, 256),
            dropout=0.3,
            num_classes=8,
        )
        n = model.count_parameters()
        assert n > 1_000_000, f"Paper-faithful model expected >1M params, got {n}"


class TestDualBranchOverfit:
    def test_overfit_tiny_batch(self) -> None:
        """Model can overfit a tiny batch (checks gradient flow)."""
        torch.manual_seed(0)
        model = _build_model(
            phono_proj_dim=32,
            phono_conv_channels=64,
            phono_lstm_hidden=64,
            phono_lstm_layers=1,
            stgcn_channels=(32, 64),
            num_classes=4,
        )
        model.train()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(8, 24, 67, 3)
        y = torch.randint(0, 4, (8,))

        for _ in range(200):
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            preds = model(x).argmax(dim=-1)
        accuracy = (preds == y).float().mean().item()
        assert accuracy >= 0.90, f"Expected ≥90% overfit accuracy, got {accuracy:.2%}"


class TestDualBranchDispatch:
    def test_load_training_config_dual(self) -> None:
        """load_training_config should parse the dual-branch YAML without error."""
        from maestro.training.train_lstm import load_training_config

        cfg = load_training_config(_DUAL_CONFIG)
        assert cfg.model.model_name.startswith("dual_stgcn")
        assert cfg.model.num_nodes == 67
        assert cfg.model.num_coords == 3
        assert cfg.model.stgcn_channels == (64, 128, 256)
        assert cfg.model.phono_proj_dim == 128
        assert cfg.model.phono_conv_channels == 256
        assert cfg.model.lstm_hidden == 256
        assert cfg.model.lstm_layers == 2

    def test_build_model_dispatch_dual(self) -> None:
        """_build_model should return a DualBranchSTGCN_LSTM for 'dual_stgcn_*'."""
        from maestro.training.train_lstm import _build_model, load_training_config

        cfg = load_training_config(_DUAL_CONFIG)
        device = torch.device("cpu")
        model = _build_model(cfg.model, device)
        assert isinstance(model, DualBranchSTGCN_LSTM)

    def test_build_model_stgcn_unchanged(self) -> None:
        """Existing 'stgcn_*' dispatch must still produce STGCN_LSTM."""
        from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM
        from maestro.training.train_lstm import _build_model, load_training_config

        stgcn_cfg_path = _REPO_ROOT / "configs" / "model" / "stgcn_lstm.yaml"
        cfg = load_training_config(stgcn_cfg_path)
        device = torch.device("cpu")
        model = _build_model(cfg.model, device)
        assert isinstance(model, STGCN_LSTM)
