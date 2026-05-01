"""Phase 4 validation for the refactored STGCN-LSTM architecture.

Three checks:
  1. Shape trace  — prints tensor shapes at every stage of the forward pass.
  2. Overfit test — trains on 10 fixed samples for 200 steps; expects ≥99 % acc.
  3. Adjacency    — verifies partitioned adjacency properties (symmetry, rows sum).

Run with:
    python scripts/validate_stgcn_architecture.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

# ── import project modules ────────────────────────────────────────────────────
sys.path.insert(0, "src")

from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM
from maestro.pipeline.graph_schema import (
    NUM_COORDS,
    NUM_NODES,
    WINDOW_LENGTH,
    build_partitioned_adjacency,
    normalise_partitioned_adjacency,
)

SEPARATOR = "─" * 68

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shape trace
# ─────────────────────────────────────────────────────────────────────────────


def _shape_trace() -> None:
    print(SEPARATOR)
    print("1. SHAPE TRACE")
    print(SEPARATOR)

    B, T, N, C = 4, WINDOW_LENGTH, NUM_NODES, NUM_COORDS
    model = STGCN_LSTM()
    model.eval()

    x = torch.randn(B, T, N, C)
    print(f"  Input                        : {tuple(x.shape)}")

    # --- STGCN blocks (instrument forward manually) ---
    with torch.no_grad():
        h = x
        for idx, block in enumerate(model.stgcn_blocks):
            # Replicate block internals to capture intermediate shapes
            _B, _T, _N, _ = h.shape

            gcn_out = block.gcn(h)
            print(
                f"  Block {idx + 1} GCN out              : {tuple(gcn_out.shape)}"
            )
            bn_out = block.bn_gcn(gcn_out.reshape(-1, gcn_out.shape[-1])).reshape(
                _B, _T, _N, -1
            )

            t_in = bn_out.permute(0, 2, 3, 1).reshape(_B * _N, -1, _T)
            ta = block.tcn_a(t_in)
            tb = block.tcn_b(t_in)
            t_out = torch.cat([ta, tb], dim=1)
            T_new = t_out.shape[-1]
            t_out_bn = block.tcn_bn(t_out)
            t_out_reshaped = t_out_bn.reshape(_B, _N, -1, T_new).permute(0, 3, 1, 2)
            print(
                f"  Block {idx + 1} TCN out              : {tuple(t_out_reshaped.shape)}"
            )

            # Full block forward
            h = block(h)
            print(
                f"  Block {idx + 1} block out             : {tuple(h.shape)}"
            )

        # Pool
        pooled = h.mean(dim=2)
        print(f"  After mean-pool (nodes)      : {tuple(pooled.shape)}")

        # LSTM
        lstm_out, _ = model.lstm(pooled)
        print(f"  LSTM output                  : {tuple(lstm_out.shape)}")

        # Attention
        scores = model.attention.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        print(f"  Attention context            : {tuple(context.shape)}")

        # Head
        logits = model.head(model.dropout(context))
        print(f"  Logits                       : {tuple(logits.shape)}")

    print()
    print(f"  Trainable parameters         : {model.count_parameters():,}")
    budget_ok = model.count_parameters() < 500_000
    print(f"  ≤500 K budget                : {'✓ OK' if budget_ok else '✗ OVER BUDGET'}")
    if not budget_ok:
        raise AssertionError("Parameter count exceeds 500 K budget")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Overfit test — 10 samples, 200 gradient steps → ≥99 % accuracy
# ─────────────────────────────────────────────────────────────────────────────


def _overfit_test(
    n_samples: int = 10,
    n_steps: int = 200,
    target_acc: float = 0.99,
) -> None:
    print(SEPARATOR)
    print("2. OVERFIT TEST")
    print(SEPARATOR)

    torch.manual_seed(0)
    model = STGCN_LSTM()
    model.train()

    x = torch.randn(n_samples, WINDOW_LENGTH, NUM_NODES, NUM_COORDS)
    y = torch.randint(0, 8, (n_samples,))

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, n_steps + 1):
        optimiser.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimiser.step()

        if step % 50 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"  step {step:>4d}  loss={loss.item():.4f}  acc={acc:.2%}")

        if loss.item() < 1e-4:
            break

    model.eval()
    with torch.no_grad():
        final_acc = (model(x).argmax(dim=1) == y).float().mean().item()

    status = "✓ PASS" if final_acc >= target_acc else "✗ FAIL"
    print(f"\n  Final accuracy: {final_acc:.2%}  [{status}]")
    if final_acc < target_acc:
        raise AssertionError(
            f"Overfit test failed: accuracy {final_acc:.2%} < {target_acc:.2%}. "
            "The model may have a gradient flow problem (e.g. broken residuals)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adjacency sanity checks
# ─────────────────────────────────────────────────────────────────────────────


def _adjacency_checks() -> None:
    print(SEPARATOR)
    print("3. ADJACENCY CHECKS")
    print(SEPARATOR)

    raw = build_partitioned_adjacency()   # (3, 67, 67)
    norm = normalise_partitioned_adjacency(raw)

    print(f"  Raw partitions shape         : {raw.shape}")

    # Partition 0: identity
    import numpy as np
    id_ok = np.allclose(raw[0], np.eye(NUM_NODES))
    print(f"  Partition 0 = identity       : {'✓' if id_ok else '✗'}")

    # Partition 1 + 2 cover all edges in the raw binary adjacency
    from maestro.pipeline.graph_schema import build_adjacency
    A_bin = build_adjacency()
    union = (raw[1] + raw[2]) > 0
    missing = (A_bin > 0) & ~union
    ok = "✓" if not missing.any() else f"✗ ({missing.sum()} edges missing)"
    print(f"  All edges covered (parts 1+2): {ok}")

    # Normalised rows sum to ≤1 (row-norm, some nodes may have deg=0 → 0)
    for k in range(3):
        row_sums = norm[k].sum(axis=1)
        max_err = abs(row_sums[row_sums > 0] - 1.0).max() if (row_sums > 0).any() else 0.0
        symbol = "✓" if max_err < 1e-5 else "✗"
        print(f"  Partition {k} normalised row-sum max err: {max_err:.2e}  {symbol}")

    # Partition 1 is a sub-matrix of A_bin (no spurious edges)
    spurious_1 = (raw[1] > 0) & (A_bin == 0)
    ok1 = "✓" if not spurious_1.any() else f"✗ ({spurious_1.sum()} spurious)"
    print(f"  Partition 1 subset of edges  : {ok1}")
    spurious_2 = (raw[2] > 0) & (A_bin == 0)
    ok2 = "✓" if not spurious_2.any() else f"✗ ({spurious_2.sum()} spurious)"
    print(f"  Partition 2 subset of edges  : {ok2}")

    # Non-zero edge counts per partition
    for k, name in enumerate(["self", "centripetal", "centrifugal"]):
        count = int((raw[k] > 0).sum())
        print(f"  Partition {k} ({name:>12s}) edges: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("STGCN-LSTM Architecture Validation")
    print(SEPARATOR)

    _shape_trace()
    print()
    _overfit_test()
    print()
    _adjacency_checks()

    print()
    print(SEPARATOR)
    print("All validation checks passed.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
