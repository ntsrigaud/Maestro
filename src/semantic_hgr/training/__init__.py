"""
Architecture A — Training Package
====================================
Training loops for the 4-phase Architecture A protocol.

Planned modules:
    train_gcn       Phase 1: GCN geometry stream standalone training
    train_rgb       Phase 2: ViT-B/16 RGB baseline training
    train_dual      Phase 3+4: Dual-stream cross-attention fusion training
    losses          Loss functions (CE + optional contrastive)
    schedulers      LR schedulers (cosine warm-up)

Each trainer is invoked via the corresponding Makefile target:
    make train-gcn | make train-rgb | make train-dual
"""
