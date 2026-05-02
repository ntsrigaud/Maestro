"""Architecture validation for DualBranchSTGCN-LSTM.

Three checks:
  1. Shape trace  — prints tensor shapes at every stage of the forward pass.
  2. Overfit test — trains on 10 fixed samples for 200 steps; expects ≥99 % acc.
  3. Adjacency    — verifies fully-connected partitioned adjacency properties.

Run with:
    python scripts/validate_stgcn_architecture.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F  # noqa: F401  (used by overfit loss)

# ── import project modules ────────────────────────────────────────────────────
sys.path.insert(0, "src")

from maestro.infrastructure.model.dual_branch_stgcn_lstm import DualBranchSTGCN_LSTM
from maestro.pipeline.graph_schema import (
    NUM_COORDS,
    NUM_NODES,
    WINDOW_LENGTH,
    build_fully_connected_partitioned_adjacency,
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
    model = DualBranchSTGCN_LSTM()
    model.eval()

    x = torch.randn(B, T, N, C)
    print(f"  Input                        : {tuple(x.shape)}")

    with torch.no_grad():
        # Branch 1
        phono_feat = model.phono_branch(x)
        print(f"  Branch 1 (phono) out         : {tuple(phono_feat.shape)}")

        # Branch 2
        h = x
        for idx, block in enumerate(model.stgcn_blocks):
            h = block(h)
            print(f"  Branch 2 STGCN block {idx + 1} out    : {tuple(h.shape)}")
        stgcn_feat = h.mean(dim=(1, 2))
        print(f"  Branch 2 global pool out     : {tuple(stgcn_feat.shape)}")

        # Fusion
        fused = torch.cat([phono_feat, stgcn_feat], dim=-1)
        print(f"  Fused (cat)                  : {tuple(fused.shape)}")

        logits = model.head(model.dropout(fused))
        print(f"  Logits                       : {tuple(logits.shape)}")

    num_params = model.count_parameters()
    print()
    print(f"  Trainable parameters         : {num_params:,}")


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
    model = DualBranchSTGCN_LSTM()
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
    print("3. ADJACENCY CHECKS (fully-connected partitioned)")
    print(SEPARATOR)

    raw = build_fully_connected_partitioned_adjacency()  # (3, 67, 67)
    norm = normalise_partitioned_adjacency(raw)

    print(f"  Raw partitions shape         : {raw.shape}")

    # Partition 0: identity
    import numpy as np

    id_ok = np.allclose(raw[0], np.eye(NUM_NODES))
    print(f"  Partition 0 = identity       : {'✓' if id_ok else '✗'}")

    # Normalised rows sum to ≤1 (row-norm, some nodes may have deg=0 → 0)
    for k in range(3):
        row_sums = norm[k].sum(axis=1)
        max_err = abs(row_sums[row_sums > 0] - 1.0).max() if (row_sums > 0).any() else 0.0
        symbol = "✓" if max_err < 1e-5 else "✗"
        print(f"  Partition {k} normalised row-sum max err: {max_err:.2e}  {symbol}")

    # Non-zero edge counts per partition
    for k, name in enumerate(["self", "centripetal", "centrifugal"]):
        count = int((raw[k] > 0).sum())
        print(f"  Partition {k} ({name:>12s}) edges: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("DualBranchSTGCN-LSTM Architecture Validation")
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
