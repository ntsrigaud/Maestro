"""
Semantic Intent Extractor - Layer 3: Semantic Intent

Maps visual embeddings to semantic intents using CLIP's text encoder
and cosine similarity matching.

Architecture Role:
- Bridges visual representation (Layer 2) to semantic meaning
- Enables zero-shot gesture understanding via text descriptions
- Provides ranked intent predictions with confidence scores

Design Principles:
- Intent vocabulary is declarative and extensible
- No retraining required for new intents (zero-shot)
- Clean separation from command execution logic
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
import torch
import open_clip

logger = logging.getLogger(__name__)


# ==============================================================================
# TIER-1: STATIC SEMANTIC INTENTS (HaGRID-Aligned, Single-Frame Recognizable)
# ==============================================================================
# Vocabulary realignment (Feb 12, 2026): Restricted to pose-based gestures
# that can be recognized from single RGB frames without temporal modeling.
#
# Design Criteria:
# 1. Recognizable from single frame (pose-based, not motion-based)
# 2. Visually distinct hand configurations
# 3. Corresponds to HaGRID dataset classes
# 4. Human-interpretable semantic meaning
#
# See: docs/analysis/INTENT_CLASSIFICATION.md for full taxonomy
# ==============================================================================

STATIC_SEMANTIC_INTENTS = {
    # Approval/Confirmation gestures
    "thumbs_up": [
        "thumbs up gesture",
        "like gesture",
        "approval sign",
        "positive hand sign",
        "thumb extended upward"
    ],
    "thumbs_down": [
        "thumbs down gesture",
        "dislike gesture",
        "rejection sign",
        "negative hand sign",
        "thumb extended downward"
    ],
    
    # Basic hand poses
    "open_palm": [
        "open palm gesture",
        "stop gesture",
        "flat hand facing forward",
        "wait sign",
        "five fingers extended"
    ],
    "closed_fist": [
        "closed fist gesture",
        "clenched hand",
        "fist pose",
        "neutral hand pose",
        "all fingers closed"
    ],
    
    # Symbolic gestures
    "peace_sign": [
        "peace sign gesture",
        "victory sign",
        "two fingers up in V shape",
        "V gesture with hand",
        "index and middle fingers extended"
    ],
    "ok_sign": [
        "ok sign gesture",
        "okay hand gesture",
        "perfect sign",
        "circle with thumb and index finger",
        "OK hand pose"
    ],
    
    # Pointing and action gestures
    "pointing": [
        "pointing gesture",
        "index finger extended pointing",
        "indicate with finger",
        "point at something",
        "one finger pointing forward"
    ],
    "call_gesture": [
        "call me gesture",
        "phone sign with hand",
        "hang loose gesture",
        "shaka sign",
        "thumb and pinky extended"
    ],
    
    # Idle/Background state
    "no_hand": [
        "no hand detected",
        "empty frame",
        "no gesture present",
        "background without hand",
        "idle state no hand"
    ]
}


# ==============================================================================
# LEGACY: ORIGINAL MIXED VOCABULARY (DEPRECATED FOR STATIC-ONLY MODE)
# ==============================================================================
# Contains 13 dynamic intents (68%) that require temporal modeling.
# Preserved for backward compatibility but NOT RECOMMENDED for production use.
#
# CRITICAL: Dynamic intents (scroll, swipe, zoom, etc.) cannot be reliably
# recognized from single frames. See root cause analysis:
# docs/analysis/CALIBRATION_ROOT_CAUSE_ANALYSIS.md
# ==============================================================================

SEMANTIC_INTENTS = {
    # Navigation (DYNAMIC - requires motion trajectories)
    "scroll_up": [
        "scroll up",
        "move up",
        "go higher",
        "scroll upward",
        "page up"
    ],
    "scroll_down": [
        "scroll down",
        "move down",
        "go lower",
        "scroll downward",
        "page down"
    ],
    "swipe_left": [
        "swipe left",
        "move left",
        "go to previous",
        "navigate left"
    ],
    "swipe_right": [
        "swipe right",
        "move right",
        "go to next",
        "navigate right"
    ],
    
    # Volume control (DYNAMIC - semantic actions without clear poses)
    "increase_volume": [
        "increase volume",
        "turn up sound",
        "make it louder",
        "volume up",
        "raise volume"
    ],
    "decrease_volume": [
        "decrease volume",
        "turn down sound",
        "make it quieter",
        "volume down",
        "lower volume"
    ],
    "mute": [
        "mute",
        "silence",
        "turn off sound",
        "mute audio"
    ],
    
    # Selection and control (AMBIGUOUS - no clear visual grounding)
    "select": [
        "select this",
        "choose this",
        "click here",
        "confirm selection",
        "pick this"
    ],
    "cancel": [
        "cancel",
        "dismiss",
        "close this",
        "go back",
        "reject"
    ],
    "confirm": [
        "confirm",
        "accept",
        "okay",
        "yes",
        "approve"
    ],
    
    # Playback control (AMBIGUOUS - semantic actions)
    "play": [
        "play",
        "start playback",
        "resume",
        "unpause"
    ],
    "pause": [
        "pause",
        "stop temporarily",
        "halt playback"
    ],
    "stop": [
        "stop",
        "end playback",
        "terminate"
    ],
    
    # Zoom and focus (DYNAMIC - requires temporal tracking)
    "zoom_in": [
        "zoom in",
        "magnify",
        "enlarge",
        "make bigger",
        "zoom closer"
    ],
    "zoom_out": [
        "zoom out",
        "shrink",
        "make smaller",
        "zoom further"
    ],
    
    # Generic actions (STATIC - kept for backward compatibility)
    "point": [
        "point at",
        "indicate",
        "show",
        "highlight"
    ],
    "grab": [
        "grab",
        "hold",
        "grasp",
        "pick up"
    ],
    "release": [
        "release",
        "let go",
        "drop",
        "put down"
    ],
    
    # No gesture (STATIC - idle state)
    "no_gesture": [
        "no gesture",
        "idle",
        "waiting",
        "resting",
        "no action"
    ]
}

# Dynamic intents that require temporal modeling (INVALID for single-frame inference)
DYNAMIC_INTENTS = [
    "scroll_up", "scroll_down", "swipe_left", "swipe_right",
    "zoom_in", "zoom_out", "drag",
    "increase_volume", "decrease_volume", "mute",
    "play", "pause", "stop"
]

# Ambiguous intents without clear visual grounding
AMBIGUOUS_INTENTS = ["select", "cancel", "confirm"]


class SemanticIntentExtractor:
    """
    Extracts semantic intent from visual embeddings using CLIP text encoder.
    
    Uses CLIP's joint vision-language space to match gesture embeddings
    against predefined semantic intent descriptions via cosine similarity.
    
    Correctness Notes (updated Mar 2, 2026):
    - CLIP's logit_scale is the sole temperature mechanism per Radford et al. 2021.
      Set temperature=1.0 (default) when apply_logit_scale=True.
    - Combining logit_scale (~100) with temperature=0.07 produces ~1428× total
      scaling, collapsing softmax to a degenerate one-hot distribution and
      causing catastrophic overconfidence (ECE ~0.857 observed Feb 2026).
    - no_hand is excluded from the CLIP vocabulary in static_only mode; MediaPipe
      gate at Layer 1 already handles the no-hand case before this layer is reached.
    
    Architecture Boundary:
    - Input: NumPy array (embedding_dim,) from CLIPEncoder
    - Output: List of (intent_name, confidence) tuples
    
    Example:
        >>> from representation import CLIPEncoder
        >>> encoder = CLIPEncoder()
        >>> extractor = SemanticIntentExtractor(
        ...     encoder.model,
        ...     encoder.device,
        ...     SEMANTIC_INTENTS
        ... )
        >>> gesture_embedding = encoder.encode_image(hand_image)
        >>> intents = extractor.extract_intent(gesture_embedding, top_k=3)
        >>> intents[0]
        ('increase_volume', 0.87)
    """
    
    def __init__(
        self,
        clip_model,
        device: str,
        intent_vocabulary: Dict[str, List[str]] = None,
        use_prompt_template: bool = True,
        temperature: float = 1.0,
        apply_logit_scale: bool = True,
        static_only: bool = True
    ):
        """
        Initialize semantic intent extractor.
        
        Args:
            clip_model: Loaded CLIP model (shared with CLIPEncoder)
            device: Compute device ('cpu', 'cuda', 'mps')
            intent_vocabulary: Dict mapping intent names to text descriptions
                If None, uses STATIC_SEMANTIC_INTENTS (static_only=True) or
                SEMANTIC_INTENTS (static_only=False, legacy mode)
            use_prompt_template: Whether to use structured prompt templates
                (e.g., "a photo of a hand performing {action}")
            temperature: Additional divisor applied to logits after logit_scale.
                Default: 1.0 (identity — no additional scaling beyond logit_scale).
                IMPORTANT: Do not set below 1.0 when apply_logit_scale=True.
                CLIP's logit_scale (~100) is already the temperature inverse from
                pretraining (Radford et al. 2021, Eq. 1). Dividing again by 0.07
                yields ~1428× total scaling and forces all output distributions to
                near one-hot, producing ECE > 0.85 and 4% accuracy.
                Only tune this below 1.0 when apply_logit_scale=False.
            apply_logit_scale: Whether to apply CLIP's learned logit_scale.
                Default True matches CLIP's standard zero-shot formulation.
            static_only: Whether to restrict to static gesture vocabulary (default: True)
                When True, uses STATIC_SEMANTIC_INTENTS minus the no_hand class
                (MediaPipe at Layer 1 already handles no-hand gating).
                When False, uses legacy SEMANTIC_INTENTS (19 intents, includes dynamic)
        
        Raises:
            ValueError: If intent_vocabulary is empty
        """
        # Set vocabulary based on mode
        if intent_vocabulary is None:
            if static_only:
                self.intent_vocabulary = STATIC_SEMANTIC_INTENTS
                logger.info("Using STATIC_SEMANTIC_INTENTS (9 HaGRID-aligned poses)")
            else:
                self.intent_vocabulary = SEMANTIC_INTENTS
                logger.warning(
                    "Using legacy SEMANTIC_INTENTS (19 intents). "
                    "WARNING: 68% are dynamic gestures invalid for single-frame inference. "
                    "Set static_only=True for production use."
                )
        else:
            self.intent_vocabulary = intent_vocabulary
        
        # Validate vocabulary is not empty
        if not self.intent_vocabulary:
            raise ValueError("Intent vocabulary cannot be empty")
        
        # In static_only mode MediaPipe (Layer 1) already gates the no-hand case
        # before any crop reaches the CLIP encoder. Including no_hand in the
        # closed-set softmax consumes probability mass from the 8 valid gesture
        # classes and produces semantically incoherent confidence values.
        if static_only and 'no_hand' in self.intent_vocabulary:
            self.intent_vocabulary = {
                k: v for k, v in self.intent_vocabulary.items() if k != 'no_hand'
            }
            logger.info(
                "Removed 'no_hand' from CLIP classification vocabulary "
                "(static_only=True: MediaPipe gates no-hand upstream at Layer 1)"
            )
        
        self.clip_model = clip_model
        self.device = device
        self.use_prompt_template = use_prompt_template
        self.temperature = temperature
        self.apply_logit_scale = apply_logit_scale
        self.static_only = static_only
        
        # Extract CLIP logit_scale parameter (critical for correct similarities)
        if hasattr(clip_model, 'logit_scale'):
            self.logit_scale = clip_model.logit_scale.exp().item()
        else:
            logger.warning("CLIP model has no logit_scale parameter. Using 1.0.")
            self.logit_scale = 1.0
        
        logger.info(
            f"Initializing SemanticIntentExtractor with "
            f"{len(self.intent_vocabulary)} intents, "
            f"prompt_template={use_prompt_template}, "
            f"temperature={temperature:.3f}, "
            f"logit_scale={self.logit_scale:.2f}"
        )
        
        # Precompute text embeddings for all intent descriptions
        self._precompute_text_embeddings()
        
        # Calculate total number of text variations
        total_descriptions = sum(len(variations) for variations in self.intent_vocabulary.values())
        
        logger.info(
            f"SemanticIntentExtractor initialized: "
            f"{len(self.intent_names)} intents, "
            f"{total_descriptions} text descriptions"
        )
    
    def _precompute_text_embeddings(self):
        """
        Precompute and cache text embeddings for all intent descriptions.
        
        Converts all text variations in intent vocabulary to CLIP text
        embeddings and averages them per intent for robust matching.
        
        IMPROVEMENT: Optionally applies structured prompt templates for
        better semantic alignment with CLIP's training data.
        """
        self.intent_names = []
        self.text_embeddings_list = []
        
        # Tokenize and encode all text descriptions
        for intent_name, text_variations in self.intent_vocabulary.items():
            # Apply prompt template if enabled
            if self.use_prompt_template:
                # Template: "a photo of a hand performing {action}"
                # This aligns with CLIP's training distribution
                templated_variations = [
                    f"a photo of a hand performing {variation}"
                    for variation in text_variations
                ]
            else:
                templated_variations = text_variations
            
            # Tokenize text variations
            text_tokens = open_clip.tokenize(templated_variations).to(self.device)
            
            with torch.no_grad():
                # Encode to text embeddings
                text_features = self.clip_model.encode_text(text_tokens)
                
                # L2 normalize each embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Average across variations for robust representation
                intent_embedding = text_features.mean(dim=0)
                
                # Normalize the averaged embedding
                intent_embedding = intent_embedding / intent_embedding.norm()
            
            self.intent_names.append(intent_name)
            self.text_embeddings_list.append(intent_embedding.cpu().numpy())
        
        # Stack into matrix for batch similarity computation
        self.text_embeddings = np.stack(self.text_embeddings_list, axis=0).astype(np.float32)
        
        # Validate
        assert self.text_embeddings.shape[0] == len(self.intent_names), \
            "Mismatch between intent names and embeddings"
        assert np.isfinite(self.text_embeddings).all(), \
            "Text embeddings contain NaN or Inf"
    
    def extract_intent(
        self,
        gesture_embedding: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Extract semantic intents from gesture embedding.
        
        Computes cosine similarity between gesture embedding and all
        intent text embeddings, returning top-K matches.
        
        Args:
            gesture_embedding: L2-normalized embedding from CLIPEncoder
                - Shape: (embedding_dim,)
                - Dtype: float32
            top_k: Number of top intents to return (default: 5)
            min_confidence: Minimum confidence threshold (default: 0.0)
                Intents below this similarity score are filtered out
        
        Returns:
            List of (intent_name, confidence) tuples, sorted by confidence
            descending. Confidence is cosine similarity in [-1, 1].
            
        Raises:
            ValueError: If gesture_embedding has invalid shape or dtype
        
        Example:
            >>> embedding = np.random.randn(512).astype(np.float32)
            >>> embedding /= np.linalg.norm(embedding)  # normalize
            >>> intents = extractor.extract_intent(embedding, top_k=3)
            >>> len(intents) <= 3
            True
            >>> intents[0][1] >= intents[1][1]  # sorted descending
            True
        """
        # Validate input
        if not isinstance(gesture_embedding, np.ndarray):
            raise ValueError(
                f"Expected NumPy array, got {type(gesture_embedding).__name__}"
            )
        
        if gesture_embedding.ndim != 1:
            raise ValueError(
                f"Expected 1D embedding, got shape {gesture_embedding.shape}"
            )
        
        expected_dim = self.text_embeddings.shape[1]
        if gesture_embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {gesture_embedding.shape[0]}"
            )
        
        # Ensure embedding is normalized (should be, but verify)
        norm = np.linalg.norm(gesture_embedding)
        if norm < 1e-10:
            raise ValueError(
                f"Gesture embedding appears to be a zero vector (norm={norm:.2e}). "
                f"Cannot normalize a zero-length vector."
            )
        if not np.isclose(norm, 1.0, atol=1e-5):
            logger.warning(
                f"Gesture embedding not normalized (norm={norm:.4f}). "
                f"Normalizing automatically."
            )
            gesture_embedding = gesture_embedding / norm
        
        # Compute cosine similarities via dot product (embeddings are normalized)
        similarities = np.dot(self.text_embeddings, gesture_embedding)
        
        # CRITICAL FIX: Apply CLIP logit_scale
        # CLIP learns a temperature parameter (logit_scale) during training
        # that scales similarities to the correct range. Without this, 
        # confidences are ~100x too low!
        if self.apply_logit_scale:
            similarities = similarities * self.logit_scale
        
        # Apply temperature scaling for sharper distributions
        # Lower temperature = more confident predictions
        # CLIP default: 0.07 for zero-shot classification
        similarities = similarities / self.temperature
        
        # Validate logits before softmax
        if not np.isfinite(similarities).all():
            raise RuntimeError("Logit computation produced NaN or Inf")
        
        # CRITICAL: Apply softmax to convert logits to probabilities
        # This step is REQUIRED for confidence values to be valid probabilities [0,1]
        # Reference: Radford et al., 2021, "Learning Transferable Visual Models..."
        # Formula: P(class_i) = exp(logit_i) / sum(exp(logit_j) for all j)
        exp_logits = np.exp(similarities - similarities.max())  # numerical stability
        probabilities = exp_logits / exp_logits.sum()
        
        # Validate probabilities
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Softmax computation produced NaN or Inf")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise RuntimeError(f"Probabilities do not sum to 1.0: {probabilities.sum():.6f}")
        
        # Use probabilities for filtering and ranking
        similarities = probabilities
        
        # Filter by minimum confidence
        valid_mask = similarities >= min_confidence
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_mask]
        
        # Sort by similarity descending
        sorted_indices = np.argsort(valid_similarities)[::-1]
        
        # Select top-K
        top_indices = sorted_indices[:top_k]
        
        # Build result list
        results = [
            (self.intent_names[valid_indices[i]], float(valid_similarities[i]))
            for i in top_indices
        ]
        
        # RUNTIME ENFORCEMENT: Filter dynamic intents in static-only mode
        if self.static_only and hasattr(self, 'static_only'):
            filtered_results = []
            for intent_name, confidence in results:
                if intent_name in DYNAMIC_INTENTS or intent_name in AMBIGUOUS_INTENTS:
                    logger.warning(
                        f"Rejected dynamic/ambiguous intent '{intent_name}' "
                        f"(conf={confidence:.3f}) in static-only mode"
                    )
                else:
                    filtered_results.append((intent_name, confidence))
            results = filtered_results
        
        logger.debug(
            f"Extracted {len(results)} intents (top-{top_k}): "
            f"{[(name, f'{conf:.3f}') for name, conf in results[:3]]}"
        )
        
        return results
    
    def extract_batch(
        self,
        gesture_embeddings: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.0
    ) -> List[List[Tuple[str, float]]]:
        """
        Extract intents for batch of gesture embeddings.
        
        More efficient than calling extract_intent() repeatedly.
        
        Args:
            gesture_embeddings: Batch of embeddings from CLIPEncoder
                - Shape: (N, embedding_dim)
                - Dtype: float32
            top_k: Number of top intents per gesture (default: 5)
            min_confidence: Minimum confidence threshold (default: 0.0)
        
        Returns:
            List of intent lists, one per gesture in batch
            
        Raises:
            ValueError: If gesture_embeddings have invalid shape or dtype
        
        Example:
            >>> embeddings = np.random.randn(3, 512).astype(np.float32)
            >>> embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            >>> batch_intents = extractor.extract_batch(embeddings, top_k=3)
            >>> len(batch_intents)
            3
            >>> len(batch_intents[0]) <= 3
            True
        """
        # Validate input
        if not isinstance(gesture_embeddings, np.ndarray):
            raise ValueError(
                f"Expected NumPy array, got {type(gesture_embeddings).__name__}"
            )
        
        if gesture_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D batch of embeddings, got shape {gesture_embeddings.shape}"
            )
        
        batch_size = gesture_embeddings.shape[0]
        expected_dim = self.text_embeddings.shape[1]
        
        if gesture_embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {gesture_embeddings.shape[1]}"
            )
        
        # Ensure embeddings are normalized
        norms = np.linalg.norm(gesture_embeddings, axis=1)
        zero_mask = norms < 1e-10
        if np.any(zero_mask):
            zero_indices = np.where(zero_mask)[0]
            raise ValueError(
                f"Batch contains {len(zero_indices)} zero vector(s) "
                f"at indices {zero_indices.tolist()}. Cannot normalize zero-length vectors."
            )
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Batch embeddings not normalized. Normalizing automatically.")
            gesture_embeddings = gesture_embeddings / norms[:, np.newaxis]
        
        # Compute cosine similarities: (N, num_intents)
        logits = np.dot(gesture_embeddings, self.text_embeddings.T)
        
        # CRITICAL FIX: Apply CLIP logit_scale and temperature
        if self.apply_logit_scale:
            logits = logits * self.logit_scale
        logits = logits / self.temperature
        
        # Apply softmax per sample to convert logits to probabilities
        # For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Validate probabilities
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Batch softmax computation produced NaN or Inf")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Batch probabilities do not sum to 1.0 for all samples")
        
        # Process each gesture
        results = []
        for i in range(batch_size):
            gesture_probabilities = probabilities[i]
            
            # Filter by minimum confidence
            valid_mask = gesture_probabilities >= min_confidence
            valid_indices = np.where(valid_mask)[0]
            valid_probabilities = gesture_probabilities[valid_mask]
            
            # Sort by probability descending
            sorted_indices = np.argsort(valid_probabilities)[::-1]
            
            # Select top-K
            top_indices = sorted_indices[:top_k]
            
            # Build result list
            gesture_intents = [
                (self.intent_names[valid_indices[j]], float(valid_probabilities[j]))
                for j in top_indices
            ]
            
            results.append(gesture_intents)
        
        return results
    
    def get_intent_names(self) -> List[str]:
        """
        Get list of all available intent names.
        
        Returns:
            List of intent names in vocabulary
        """
        return self.intent_names.copy()
    
    def get_intent_count(self) -> int:
        """
        Get number of intents in vocabulary.
        
        Returns:
            Number of distinct intents
        """
        return len(self.intent_names)
    
    def get_config(self) -> dict:
        """
        Get extractor configuration for serialization/logging.
        
        Returns:
            Dictionary containing extractor configuration
        """
        return {
            'device': self.device,
            'intent_count': len(self.intent_names),
            'embedding_dim': self.text_embeddings.shape[1],
            'intent_names': self.intent_names,
            'use_prompt_template': self.use_prompt_template,
            'temperature': self.temperature,
            'logit_scale': self.logit_scale,
            'apply_logit_scale': self.apply_logit_scale
        }
    
    def __repr__(self) -> str:
        return (
            f"SemanticIntentExtractor(intents={len(self.intent_names)}, "
            f"device='{self.device}', temp={self.temperature:.3f}, "
            f"logit_scale={self.logit_scale:.2f})"
        )
