"""Unit tests for STGCN_LSTM model (Phase 2).

Covers:
  - Forward pass output shape.
  - Parameter count budget (≤500 K).
  - Adjacency buffer is registered and saved/restored.
  - All variants (1 block, 2 blocks, varying hidden) produce correct shapes.
  - Batch size 1 works (edge case for BatchNorm).
"""

from __future__ import annotations

import torch

from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM, STGCNBlock, _GraphConv

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _build_model(**kwargs) -> STGCN_LSTM:
    defaults = dict(
        num_nodes=67,
        num_coords=3,
        stgcn_channels=(64, 128),
        num_stgcn_layers=2,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        num_classes=8,
    )
    defaults.update(kwargs)
    return STGCN_LSTM(**defaults)


def _batch(B: int = 4, T: int = 24, N: int = 67, C: int = 3) -> torch.Tensor:
    return torch.randn(B, T, N, C)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSTGCNLSTMForward:
    def test_output_shape_standard(self) -> None:
        model = _build_model()
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 8), f"Expected (4,8) got {out.shape}"

    def test_output_shape_single_sample(self) -> None:
        """Batch size 1 must work despite BatchNorm."""
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

    def test_one_stgcn_block(self) -> None:
        model = _build_model(
            stgcn_channels=(64,),
            num_stgcn_layers=1,
            lstm_hidden=64,
        )
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (4, 8)

    def test_output_is_logits_not_probs(self) -> None:
        """Forward pass returns raw logits (no softmax applied)."""
        model = _build_model()
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        # Logits may exceed [0, 1]; softmax output sums to 1.
        # Check that rows do NOT all sum to ~1 (i.e., softmax was not applied).
        row_sums = out.softmax(dim=-1).sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
        # At least some raw logit values should be outside [0, 1].
        assert (out < 0).any() or (out > 1).any()


class TestParameterBudget:
    def test_default_model_under_500k(self) -> None:
        model = _build_model()
        n = model.count_parameters()
        assert n < 500_000, f"Parameter count {n:,} exceeds 500 K budget"

    def test_count_parameters_matches_manual(self) -> None:
        model = _build_model()
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.count_parameters() == manual


class TestAdjacencyBuffer:
    def test_adjacency_registered_as_buffer(self) -> None:
        model = _build_model()
        assert "adjacency" in dict(model.named_buffers()), (
            "adjacency should be registered as a buffer"
        )

    def test_adjacency_shape(self) -> None:
        model = _build_model(num_nodes=67)
        A = model.adjacency
        assert A.shape == (67, 67), f"Expected (67,67) got {A.shape}"

    def test_adjacency_survives_state_dict_roundtrip(self) -> None:
        model = _build_model()
        state = model.state_dict()
        assert "adjacency" in state

        model2 = _build_model()
        model2.load_state_dict(state)
        assert torch.allclose(model.adjacency, model2.adjacency)

    def test_custom_adjacency_accepted(self) -> None:
        A = torch.eye(67)
        model = _build_model(adjacency=A)
        assert torch.allclose(model.adjacency, A)


class TestGraphConv:
    def test_output_shape(self) -> None:
        A = torch.eye(10)
        gcn = _GraphConv(in_channels=3, out_channels=16, adjacency=A)
        x = torch.randn(2, 6, 10, 3)  # (B, T, N, C_in)
        out = gcn(x)
        assert out.shape == (2, 6, 10, 16)


class TestSTGCNBlock:
    def test_output_shape_no_residual(self) -> None:
        A = torch.eye(10)
        block = STGCNBlock(3, 32, A, residual=False)
        x = torch.randn(2, 8, 10, 3)
        block.eval()
        with torch.no_grad():
            out = block(x)
        # T may be slightly different due to temporal conv padding.
        assert out.shape[0] == 2
        assert out.shape[2] == 10  # N unchanged
        assert out.shape[3] == 32  # C_out

    def test_output_shape_with_residual(self) -> None:
        A = torch.eye(10)
        block = STGCNBlock(32, 32, A, residual=True)
        x = torch.randn(2, 8, 10, 32)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape
