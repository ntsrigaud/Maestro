#!/usr/bin/env python3
"""
Architecture A — Installation Verification Script
==================================================
Verify that all dependencies required for the dual-stream supervised
gesture recognition system (GCN + ViT-B/16) are correctly installed.

Run after:  pip install -e .
            (or: conda env create -f environment.yml && pip install -e .)

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

import sys


def check(label: str, fn) -> bool:
    """Run a single check and print a pass/fail line."""
    try:
        fn()
        print(f"  OK  {label}")
        return True
    except Exception as exc:
        print(f"  FAIL  {label}  —  {exc}")
        return False


def main() -> int:
    print()
    print("=" * 64)
    print("Semantic HGR — Architecture A  Dependency Verification")
    print("=" * 64)
    print()

    failures = 0

    # ------------------------------------------------------------------
    # 1. Core numeric / ML stack
    # ------------------------------------------------------------------
    print("[1/5] Core ML stack")

    def _torch():
        import torch
        assert torch.__version__, "torch version missing"
        print(f"        torch {torch.__version__}", end="")

    def _numpy():
        import numpy as np
        print(f"        numpy {np.__version__}", end="")

    def _sklearn():
        import sklearn
        print(f"        scikit-learn {sklearn.__version__}", end="")

    for label, fn in [
        ("torch", _torch),
        ("numpy", _numpy),
        ("scikit-learn", _sklearn),
    ]:
        if not check(label, fn):
            failures += 1

    # ------------------------------------------------------------------
    # 2. Appearance stream — ViT-B/16 via timm
    # ------------------------------------------------------------------
    print()
    print("[2/5] Appearance stream  (timm / ViT-B/16)")

    def _timm_import():
        import timm
        print(f"        timm {timm.__version__}", end="")

    def _vit():
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=8)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"        ViT-B/16  {n_params:.1f}M params  num_classes=8", end="")

    for label, fn in [
        ("timm installed", _timm_import),
        ("ViT-B/16 instantiation", _vit),
    ]:
        if not check(label, fn):
            failures += 1

    # ------------------------------------------------------------------
    # 3. Geometry stream — GCN via torch-geometric
    # ------------------------------------------------------------------
    print()
    print("[3/5] Geometry stream  (torch-geometric / GCNConv)")

    def _pyg():
        import torch_geometric
        print(f"        torch-geometric {torch_geometric.__version__}", end="")

    def _gcnconv():
        from torch_geometric.nn import GCNConv
        layer = GCNConv(in_channels=3, out_channels=64)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"        GCNConv(3→64)  {n_params} params", end="")

    for label, fn in [
        ("torch-geometric installed", _pyg),
        ("GCNConv(3, 64) instantiation", _gcnconv),
    ]:
        if not check(label, fn):
            failures += 1

    # ------------------------------------------------------------------
    # 4. Hand keypoint extraction — MediaPipe
    # ------------------------------------------------------------------
    print()
    print("[4/5] Hand keypoint extraction  (MediaPipe + OpenCV)")

    def _mediapipe():
        import mediapipe as mp
        print(f"        mediapipe {mp.__version__}", end="")

    def _cv2():
        import cv2
        print(f"        opencv-python {cv2.__version__}", end="")

    def _hand_detector():
        from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
        # Instantiate without camera
        print(f"        HandDetector importable", end="")

    for label, fn in [
        ("mediapipe", _mediapipe),
        ("opencv-python", _cv2),
        ("HandDetector import", _hand_detector),
    ]:
        if not check(label, fn):
            failures += 1

    # ------------------------------------------------------------------
    # 5. Reporting / utilities
    # ------------------------------------------------------------------
    print()
    print("[5/5] Reporting utilities")

    def _pandas():
        import pandas as pd
        print(f"        pandas {pd.__version__}", end="")

    def _matplotlib():
        import matplotlib
        print(f"        matplotlib {matplotlib.__version__}", end="")

    def _seaborn():
        import seaborn as sns
        print(f"        seaborn {sns.__version__}", end="")

    for label, fn in [
        ("pandas", _pandas),
        ("matplotlib", _matplotlib),
        ("seaborn", _seaborn),
    ]:
        if not check(label, fn):
            failures += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    if failures == 0:
        print("Architecture A dependencies: ALL OK")
        print()
        print("Next steps:")
        print("  make extract-keypoints    # offline keypoint extraction")
        print("  make validate-keypoints   # pre-training gate (Section 2.6)")
        print("  make train-gcn            # Phase 1 GCN standalone")
    else:
        print(f"FAILED: {failures} check(s) did not pass.")
        print()
        print("Troubleshooting:")
        print("  conda env create -f environment.yml")
        print("  conda activate gesture")
        print("  pip install -e .")
    print("=" * 64)
    print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
