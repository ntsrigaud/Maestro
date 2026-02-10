"""
Semantic Intent Layer (Layer 3)

Extracts semantic meaning from visual representations using text-vision alignment.
Maps gesture embeddings to human-interpretable semantic intents.

Architecture Compliance:
- Input: 512-dim embeddings from Layer 2 (Representation)
- Output: Ranked semantic intents with confidence scores
- No direct dependency on Layer 4 (Command Resolution)
"""

from .clip_intent import SemanticIntentExtractor, SEMANTIC_INTENTS

__all__ = ['SemanticIntentExtractor', 'SEMANTIC_INTENTS']
