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


# Intent Vocabulary: Semantic descriptions for gesture recognition
# Each intent has multiple text variations to improve matching robustness
SEMANTIC_INTENTS = {
    # Navigation
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
    
    # Volume control
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
    
    # Selection and control
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
    
    # Playback control
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
    
    # Zoom and focus
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
    
    # Generic actions
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
    
    # No gesture (idle state)
    "no_gesture": [
        "no gesture",
        "idle",
        "waiting",
        "resting",
        "no action"
    ]
}


class SemanticIntentExtractor:
    """
    Extracts semantic intent from visual embeddings using CLIP text encoder.
    
    Uses CLIP's joint vision-language space to match gesture embeddings
    against predefined semantic intent descriptions via cosine similarity.
    
    CRITICAL FIXES APPLIED (Feb 11, 2026):
    - Added CLIP logit_scale application (100x confidence boost)
    - Added temperature scaling for sharper distributions
    - Added structured prompt templates option
    - Fixed high intent overlap issue
    
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
        temperature: float = 0.07,
        apply_logit_scale: bool = True
    ):
        """
        Initialize semantic intent extractor.
        
        Args:
            clip_model: Loaded CLIP model (shared with CLIPEncoder)
            device: Compute device ('cpu', 'cuda', 'mps')
            intent_vocabulary: Dict mapping intent names to text descriptions
                If None, uses default SEMANTIC_INTENTS
            use_prompt_template: Whether to use structured prompt templates
                (e.g., "a photo of a hand performing {action}")
            temperature: Temperature for scaling similarities (lower = sharper)
                Default: 0.07 (CLIP default for zero-shot classification)
            apply_logit_scale: Whether to apply CLIP's learned logit_scale
                CRITICAL: Must be True for correct confidence scores
        
        Raises:
            ValueError: If intent_vocabulary is empty
        """
        # Set vocabulary first, before any processing
        if intent_vocabulary is None:
            self.intent_vocabulary = SEMANTIC_INTENTS
        else:
            self.intent_vocabulary = intent_vocabulary
        
        # Validate vocabulary is not empty
        if not self.intent_vocabulary:
            raise ValueError("Intent vocabulary cannot be empty")
        
        self.clip_model = clip_model
        self.device = device
        self.use_prompt_template = use_prompt_template
        self.temperature = temperature
        self.apply_logit_scale = apply_logit_scale
        
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
