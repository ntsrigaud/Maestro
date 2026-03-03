"""
Semantic HGR System - Hand Gesture Recognition with Semantic Understanding

A production-grade system for recognizing hand gestures and mapping them
to semantic intents and computer commands using vision-language models.

Architecture:
    Layer 1: Perception (Input Manager + Hand Detection)
    Layer 2: Representation (CLIP Visual Encoder)
    Layer 3: Semantic Intent (CLIP Text Matching)
    Layer 4: Command Resolution (Intent → Command Mapping)
    Layer 5: Execution (OS Command Invocation)

Usage:
    from semantic_hgr.representation.clip_encoder import CLIPEncoder
    from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor
"""

__version__ = "0.1.0"
__author__ = "CSU Capstone Team"

# Expose key components at package level for convenience
from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor

__all__ = [
    "CLIPEncoder",
    "SemanticIntentExtractor",
]
