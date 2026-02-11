"""
Representation Layer (Layer 2)

Extracts visual features aligned with semantic space using CLIP.
Transforms raw hand images into semantically meaningful embeddings.

Architecture Compliance:
- Input: 224×224 RGB hand images from Layer 1 (Perception)
- Output: 512-dim embeddings aligned with CLIP text space
- No direct dependency on Layer 3 (Semantic Intent)
"""

from .clip_encoder import CLIPEncoder

# Alias for backward compatibility with scripts
GestureEncoder = CLIPEncoder

__all__ = ['CLIPEncoder', 'GestureEncoder']
