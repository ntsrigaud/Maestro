"""
Semantic HGR System — Architecture A: Dual-Stream Supervised Gesture Recognition

A supervised hand gesture recognition system combining a Graph Convolutional
Network (GCN) over MediaPipe 21-keypoint pose graphs with a fine-tuned ViT-B/16
RGB stream, fused via cross-attention, for static gesture classification.

Architecture:
    Layer 1: Perception — MediaPipe Hands (21 keypoints + bounding box)
    Layer 2: Geometric Stream — 3-layer GCN over MediaPipe bone graph → f_geo ∈ ℝ^256
    Layer 3: RGB Stream — ViT-B/16 (timm, ImageNet init, fine-tuned) → f_rgb ∈ ℝ^768
    Layer 4: Fusion — Cross-attention (Q=f_rgb, K=V=f_geo) → f_fused ∈ ℝ^512
    Layer 5: Head — MLP 512→256→8 class logits

Research target:
    H1: Acc@1(Dual) − Acc@1(RGB) ≥ 2%, p < 0.05 (McNemar's test)
    ECE ≤ 0.10 after temperature calibration

Reference: docs/research/ARCHITECTURE_A_IMPLEMENTATION_BLUEPRINT.md
"""

__version__ = "0.2.0"
__author__ = "Neil Chang — Colorado State University"

# Core perception components (stable, no ML framework dependency)
from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.recognition_engine.preprocessing import center_crop_frame

__all__ = [
    "HandDetectionResult",
    "HandDetector",
    "center_crop_frame",
]
