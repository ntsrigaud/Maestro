"""
Architecture A — Evaluation Package
=======================================
Evaluation utilities for the dual-stream supervised recognition system.

Planned modules:
    evaluate        Full evaluation runner (Top-K, ECE, McNemar's test)
    metrics         Top-K accuracy, ECE, per-class breakdown
    confusion       Confusion matrix generation and plotting
    mcnemar         Paired significance test (Architecture A vs. CLIP baseline)

Invoke via:
    make eval
    python -m semantic_hgr.evaluation.evaluate --checkpoint checkpoints/dual/best.pt
"""
