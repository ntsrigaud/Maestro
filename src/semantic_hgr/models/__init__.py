"""
Architecture A — Models Package
=================================
Dual-stream supervised gesture recognition models.

Planned modules:
    gcn_stream      GCN geometry stream (21-keypoint graph → 8-class logits)
    rgb_stream      ViT-B/16 appearance stream (224×224 crops → 8-class logits)
    dual_stream     Cross-attention fusion model (GCN + ViT-B/16)
    fusion          Cross-attention fusion layer
"""
