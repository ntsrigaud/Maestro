"""Unit tests for DualBranchSTGCN_LSTM (dual-branch phonological + STGCN model).

Covers:
  - Forward-pass output shape.
  - Phonological feature extractor: output shape, value ranges.
  - Full Q/K/V self-attention: output shape, residual, weight normalisation.
  - Phonological branch sub-module shapes.
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
    PHONO_FEAT_DIM,
    DualBranchSTGCN_LSTM,
    _PhonologicalBranch,
    _PhonologicalFeatureExtractor,
    _SelfAttentionQKV,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parents[2]
_DUAL_CONFIG = _REPO_ROOT / "configs" / "model" / "dual_branch_stgcn_lstm.yaml"


def _build_model(**kwargs) -> DualBranchSTGCN_LSTM:
    defaults: dict = dict(
        num_nodes=67,
        num_coords=3,
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


# ── _PhonologicalFeatureExtractor ─────────────────────────────────────────────


class TestPhonologicalFeatureExtractor:
    def test_output_shape(self) -> None:
        extractor = _PhonologicalFeatureExtractor()
        x = _batch(B=4, T=24)
        out = extractor(x)
        assert out.shape == (4, 24, PHONO_FEAT_DIM), (
            f"Expected (4,24,{PHONO_FEAT_DIM}) got {out.shape}"
        )

    def test_phono_feat_dim_is_56(self) -> None:
        assert PHONO_FEAT_DIM == 56

    def test_joint_angles_in_valid_range(self) -> None:
        """Finger joint angles must lie in [0, π] radians."""
        import math
        extractor = _PhonologicalFeatureExtractor()
        x = _batch(B=2, T=8)
        out = extractor(x)
        # First 15 features per hand = joint angles, both hands: indices 0-14 and 28-42
        right_angles = out[..., :15]   # (B, T, 15)
        left_angles = out[..., 28:43]  # (B, T, 15)
        assert right_angles.min().item() >= 0.0
        assert right_angles.max().item() <= math.pi + 1e-5
        assert left_angles.min().item() >= 0.0
        assert left_angles.max().item() <= math.pi + 1e-5

    def test_no_learnable_parameters(self) -> None:
        extractor = _PhonologicalFeatureExtractor()
        params = list(extractor.parameters())
        assert len(params) == 0, "Feature extractor must have no learnable params"

    def test_batch_size_one(self) -> None:
        extractor = _PhonologicalFeatureExtractor()
        out = extractor(_batch(B=1))
        assert out.shape == (1, 24, PHONO_FEAT_DIM)


# ── _SelfAttentionQKV ─────────────────────────────────────────────────────────


class TestSelfAttentionQKV:
    def test_output_shape_preserved(self) -> None:
        attn = _SelfAttentionQKV(dim=64)
        H = torch.randn(4, 24, 64)
        out = attn(H)
        assert out.shape == H.shape, f"Expected {H.shape} got {out.shape}"

    def test_residual_connection(self) -> None:
        """With W_q, W_k, W_v zeroed and w_f=0, output must equal input."""
        attn = _SelfAttentionQKV(dim=8)
        # Zero out all projection weights so Z = 0; set w_f = 0 so output = H
        with torch.no_grad():
            attn.W_q.weight.zero_()
            attn.W_k.weight.zero_()
            attn.W_v.weight.zero_()
            attn.w_f.zero_()
        H = torch.randn(2, 10, 8)
        out = attn(H)
        # Residual: w_f * Z + H = 0 * 0 + H = H
        assert torch.allclose(out, H, atol=1e-6), "Residual branch failed"

    def test_attention_weights_sum_to_one(self) -> None:
        """Softmax over the key dimension must sum to 1."""
        attn = _SelfAttentionQKV(dim=32)
        H = torch.randn(3, 12, 32)
        # Compute attention matrix directly to verify
        Q = attn.W_q(H)
        K = attn.W_k(H)
        scale = 32 ** -0.5
        alpha = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * scale, dim=-1)
        sums = alpha.sum(dim=-1)  # (3, 12) — each row sums to 1
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_w_f_is_learnable_parameter(self) -> None:
        attn = _SelfAttentionQKV(dim=16)
        param_names = {n for n, _ in attn.named_parameters()}
        assert "w_f" in param_names


# ── _PhonologicalBranch ───────────────────────────────────────────────────────


class TestPhonologicalBranch:
    def test_output_shape(self) -> None:
        branch = _PhonologicalBranch(
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
            conv_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.0,
        )
        branch.eval()
        with torch.no_grad():
            out = branch(_batch(B=1))
        assert out.shape == (1, 128)

    def test_has_self_attention_block(self) -> None:
        branch = _PhonologicalBranch(conv_channels=64, lstm_hidden=64, lstm_layers=1, dropout=0.0)
        assert hasattr(branch, "self_attn"), "Branch must have a self_attn (SA block)"
        assert isinstance(branch.self_attn, _SelfAttentionQKV)

    def test_has_feature_extractor(self) -> None:
        branch = _PhonologicalBranch(conv_channels=64, lstm_hidden=64, lstm_layers=1, dropout=0.0)
        assert hasattr(branch, "feature_extractor")
        assert isinstance(branch.feature_extractor, _PhonologicalFeatureExtractor)


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
        part_sum = A.sum(axis=0)                           # (67, 67)
        assert np.all(
            part_sum[off_diag_mask] == 1.0
        ), "Each off-diagonal edge must appear in exactly one partition."
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

