from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_model(
    hidden_size: int = 64,
    num_heads: int = 4,
    proj_dim: int = 64,
    num_classes: int = 10,
) -> EnhancedTwoStreamLSTM:
    return EnhancedTwoStreamLSTM(
        input_size=147,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1,
        num_classes=num_classes,
        num_heads=num_heads,
        proj_dim=proj_dim,
    )


# ── Shape tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "batch,seq,classes",
    [
        (1, 32, 8),
        (4, 32, 8),
        (1, 24, 8),  # works with other window lengths too
        (2, 32, 12),  # different num_classes
    ],
)
def test_forward_output_shape(batch: int, seq: int, classes: int) -> None:
    model = _make_model(num_classes=classes)
    x = torch.zeros(batch, seq, 147)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, classes), f"Expected ({batch}, {classes}), got {out.shape}"


def test_output_is_logits_not_probabilities() -> None:
    """Output should NOT be normalised (raw logits)."""
    model = _make_model()
    x = torch.randn(2, 32, 147)
    with torch.no_grad():
        out = model(x)
    row_sums = out.exp().sum(dim=-1)
    # If output were already softmaxed, each row would sum to ~1.
    # For raw logits this is almost never the case.
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05)


# ── Feature partition ─────────────────────────────────────────────────────────


def test_split_dimensions() -> None:
    x = torch.randn(3, 32, 147)
    pose, motion = EnhancedTwoStreamLSTM._split(x)
    assert pose.shape == (3, 32, 73), f"Pose wrong shape: {pose.shape}"
    assert motion.shape == (3, 32, 74), f"Motion wrong shape: {motion.shape}"


def test_split_covers_all_dims() -> None:
    """pose dims + motion dims must reconstruct all 147 input dims without overlap."""
    x = torch.arange(147, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, 1, 147)
    pose, motion = EnhancedTwoStreamLSTM._split(x)
    reconstructed = torch.cat([pose, motion], dim=-1).squeeze()
    assert reconstructed.shape == (147,)
    # Values: positions[0:63] + angles[134:144]
    # | spread[63:68] + traj[68:71] + vel[71:134] + wv[144:147]
    expected = torch.cat(
        [
            x[0, 0, :63],
            x[0, 0, 134:144],  # 73 pose dims
            x[0, 0, 63:134],
            x[0, 0, 144:],  # 74 motion dims
        ]
    )
    assert torch.allclose(reconstructed, expected)


# ── Architecture validation ───────────────────────────────────────────────────


def test_num_heads_divisibility_guard() -> None:
    """num_heads must evenly divide 2*hidden_size."""
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        EnhancedTwoStreamLSTM(
            input_size=147,
            hidden_size=64,  # 2×64 = 128
            num_layers=2,
            dropout=0.1,
            num_classes=10,
            num_heads=3,  # 128 % 3 != 0 → should raise
        )


def test_parameter_count_small_config() -> None:
    """Small config (h=64, proj=64): ~540 K params without cross-attention."""
    model = _make_model(hidden_size=64, proj_dim=64, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert 400_000 < n_params < 700_000, f"Unexpected param count: {n_params:,}"


def test_parameter_count_large_config() -> None:
    """Recommended config (h=96, proj=96): ~1.2 M params."""
    model = EnhancedTwoStreamLSTM(
        input_size=147,
        hidden_size=96,
        num_layers=2,
        dropout=0.35,
        num_classes=12,
        num_heads=8,
        proj_dim=96,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 900_000 < n_params < 1_500_000, f"Unexpected param count: {n_params:,}"


def test_bidirectional_lstm_output_width() -> None:
    """BiLSTM output should be 2×hidden_size per stream."""
    model = _make_model(hidden_size=64)
    # Access the underlying LSTM and verify bidirectional flag
    assert model.lstm_a.bidirectional is True
    assert model.lstm_b.bidirectional is True
    # hidden_size attribute of the LSTM is per-direction
    assert model.lstm_a.hidden_size == 64
    assert model.lstm_b.hidden_size == 64


def test_mha_embed_dim_matches_bilstm_output() -> None:
    model = _make_model(hidden_size=64, num_heads=4)
    assert model.mha_a.embed_dim == 128  # 2 × hidden_size
    assert model.mha_b.embed_dim == 128


def test_pool_ln_modules_exist() -> None:
    """pool_ln_a/b must be LayerNorm modules for post-pool normalisation."""
    model = _make_model(hidden_size=64)
    assert isinstance(model.pool_ln_a, nn.LayerNorm)
    assert isinstance(model.pool_ln_b, nn.LayerNorm)


def test_gate_modules_exist() -> None:
    """gate_a/b must be 2-layer MLP (nn.Sequential) for cross-stream gating."""
    model = _make_model(hidden_size=64, num_heads=4)
    assert isinstance(model.gate_a, nn.Sequential), (
        f"Expected nn.Sequential for gate_a, got {type(model.gate_a)}"
    )
    assert isinstance(model.gate_b, nn.Sequential), (
        f"Expected nn.Sequential for gate_b, got {type(model.gate_b)}"
    )
    assert isinstance(model.ln_cross_a, nn.LayerNorm)
    assert isinstance(model.ln_cross_b, nn.LayerNorm)


def test_gradient_flow_with_random_input() -> None:
    """All parameters must receive non-zero gradients when input is non-zero."""
    model = _make_model()
    x = torch.randn(2, 32, 147)  # random, not zeros
    logits = model(x)
    logits.sum().backward()
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not dead, f"Dead gradients with random input: {dead}"


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_all_parameters_receive_gradients() -> None:
    """With random (non-degenerate) input all parameters must be reachable."""
    model = _make_model()
    x = torch.randn(2, 32, 147)  # use random, not zeros
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not dead, f"Parameters with zero/None gradient: {dead}"


# ── Checkpoint round-trip ─────────────────────────────────────────────────────


def test_state_dict_round_trip(tmp_path: Path) -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(2, 32, 147)
    with torch.no_grad():
        out_before = model(x).clone()

    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = _make_model()
    model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model2.eval()

    with torch.no_grad():
        out_after = model2(x)

    assert torch.allclose(out_before, out_after, atol=1e-6)


# ── Checkpoint loader integration ─────────────────────────────────────────────


def test_checkpoint_loader_two_stream_attn(tmp_path: Path) -> None:
    from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

    model = _make_model()
    model.eval()

    payload = {
        "artifact_type": "two_stream_attn_inference",
        "artifact_version": "two_stream_attn_test_v1",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_name": "two_stream_attn_v1",
            "input_size": 147,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 10,
            "num_heads": 4,
            "proj_dim": 64,
        },
        "class_labels": [
            "unknown",
            "point_one",
            "point_two",
            "open_palm_hold",
            "swipe_right",
            "swipe_left",
            "swipe_up",
            "swipe_down",
            "zoom_in",
            "zoom_out",
        ],
        "feature_schema": {
            "feature_schema_version": "feature-schema-v5",
            "feature_dim": 147,
            "window_length": 32,
            "window_step": 1,
            "orientation_normalization": False,
        },
    }
    artifact_path = tmp_path / "enhanced.pt"
    torch.save(payload, artifact_path)

    artifact = load_inference_artifact(artifact_path=artifact_path, device=torch.device("cpu"))

    assert artifact.model_version == "two_stream_attn_test_v1"
    assert isinstance(artifact.model, EnhancedTwoStreamLSTM)
    assert artifact.feature_schema_version == "feature-schema-v5"
    assert artifact.window_length == 32
    assert artifact.window_step == 1
    assert len(artifact.class_labels) == 10

    # Verify inference works after loading
    x = torch.zeros(1, 32, 147)
    with torch.no_grad():
        out = artifact.model(x)
    assert out.shape == (1, 10)


# ── _build_model dispatch ─────────────────────────────────────────────────────


def test_build_model_dispatches_two_stream_attn() -> None:
    import torch

    from maestro.training.train_lstm import ModelConfig, _build_model

    cfg = ModelConfig(
        model_name="two_stream_attn_v1",
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=10,
        num_heads=4,
        proj_dim=64,
    )
    model = _build_model(cfg, torch.device("cpu"))
    assert isinstance(model, EnhancedTwoStreamLSTM)
    x = torch.zeros(1, 32, 147)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_build_model_dispatches_ipn_baseline_adapter() -> None:
    import torch

    from maestro.infrastructure.model.ipn_baseline_adapter import IPNFeatureToVideoAdapter
    from maestro.training.train_lstm import ModelConfig, _build_model

    cfg = ModelConfig(
        model_name="ipn_baseline_v1",
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=10,
        baseline_backbone="resnetl10",
        sample_size=32,
    )
    model = _build_model(cfg, torch.device("cpu"), sequence_length=16)
    assert isinstance(model, IPNFeatureToVideoAdapter)
    model.eval()
    x = torch.zeros(1, 16, 147)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)
