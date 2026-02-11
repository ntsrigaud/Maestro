# Semantic Intent Layer (Layer 3)

**Purpose**: Extract semantic meaning from visual embeddings using text-vision alignment.

## Overview

This layer maps gesture embeddings from Layer 2 (Representation) to human-interpretable semantic intents using CLIP's text encoder and cosine similarity matching. Enables zero-shot gesture understanding without training classifiers.

**Mathematical Specification:** See [CLIP Mathematical Specification](../../docs/CLIP_MATHEMATICAL_SPECIFICATION.md) for authoritative formulas and probability computation details.

**Key Properties:**

- ✅ All confidence scores are valid probabilities: P(intent) ∈ [0, 1]
- ✅ Probabilities sum to 1.0 across all intents
- ✅ Softmax normalization applied (corrected Feb 11, 2026)

## Mathematical Pipeline

**Formula:**

```
P(intent_i | gesture) = softmax((cosine_sim(g, t_i) × logit_scale) / temperature)
```

Where:

- `g` = gesture embedding (512-dim, L2-normalized)
- `t_i` = intent text embedding (512-dim, L2-normalized)
- `logit_scale ≈ 100` (extracted from CLIP model)
- `temperature = 0.07` (default for semantic intent)
- `softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)`

**Critical:** Softmax transformation is required to convert logits → probabilities. Without it, scores can exceed 1.0 (mathematically invalid).

**See:** [CLIP Mathematical Specification](../../docs/CLIP_MATHEMATICAL_SPECIFICATION.md) for complete derivation, validation, and implementation details.

## Components

### SemanticIntentExtractor

**File**: `clip_intent.py`

**Responsibility**: Match visual embeddings to semantic intent descriptions

**Architecture**:

- Input: 512-dim embeddings from CLIPEncoder
- Output: Ranked list of (intent_name, confidence) tuples
- Matching: Cosine similarity between gesture and precomputed text embeddings

**Key Features**:

- Zero-shot learning: Add intents without retraining
- Top-K ranking with confidence scores
- Batch processing support
- Deterministic matching

**Usage Example**:

```python
from representation import CLIPEncoder
from semantics import SemanticIntentExtractor, SEMANTIC_INTENTS
import numpy as np

# Initialize components
encoder = CLIPEncoder()
extractor = SemanticIntentExtractor(
    clip_model=encoder.model,
    device=encoder.device,
    intent_vocabulary=SEMANTIC_INTENTS
)

# Encode gesture and extract intent
hand_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
embedding = encoder.encode_image(hand_image)
intents = extractor.extract_intent(embedding, top_k=3)

# Result: [(intent_name, confidence), ...]
print(intents[0])  # ('scroll_up', 0.87)
```

### SEMANTIC_INTENTS Vocabulary

**File**: `clip_intent.py`

**Structure**: Dictionary mapping intent names to text description variations

**Size**: 19 intents (navigation, volume, selection, playback, zoom, actions)

**Extensibility**: Add new intents by adding text descriptions (no retraining)

**Example Intent**:

```python
"increase_volume": [
    "increase volume",
    "turn up sound",
    "make it louder",
    "volume up",
    "raise volume"
]
```

Multiple descriptions improve matching robustness through semantic averaging.

## Architecture Compliance

✅ **Clean Boundaries**: Receives embeddings from Layer 2, outputs intents to Layer 4  
✅ **Interface Contract**: NumPy embeddings in, Python tuples out  
✅ **Zero-Shot**: No training required, text-only vocabulary expansion  
✅ **Testability**: Fully covered by unit tests with mocked CLIP model

## Performance

- **Intent extraction**: ~1ms (precomputed text embeddings)
- **Vocabulary size**: 19 intents
- **Text embeddings**: Computed once at initialization
- **Memory footprint**: ~40KB (19 × 512 × 4 bytes)

## Testing

Run Layer 3 tests:

```bash
python -m unittest tests.test_semantic_intent -v
```

**Coverage**: 22 tests validating initialization, semantic matching, vocabulary, batch processing

## Intent Vocabulary

Current intents (19 total):

**Navigation**: scroll_up, scroll_down, swipe_left, swipe_right  
**Volume**: increase_volume, decrease_volume, mute  
**Selection**: select, cancel, confirm  
**Playback**: play, pause, stop  
**Zoom**: zoom_in, zoom_out  
**Actions**: point, grab, release  
**Idle**: no_gesture

## Adding New Intents

Zero-shot learning allows adding intents without retraining:

```python
# 1. Add to vocabulary
SEMANTIC_INTENTS["new_intent"] = [
    "description 1",
    "description 2",
    "description 3"
]

# 2. Reinitialize extractor
extractor = SemanticIntentExtractor(...)

# That's it! New intent is now recognizable.
```

## Future Enhancements

1. **Vocabulary Expansion**: User studies to identify additional intents
2. **Context Awareness**: Application-specific intent sets (browser vs media player)
3. **Confidence Calibration**: Optimize similarity thresholds
4. **Structured Intents**: Frame semantics with parameters (target, magnitude)

## Dependencies

- `torch>=2.0.0`
- `open-clip-torch>=2.20.0`
- `numpy>=1.24.0`

## See Also

- [Architecture Specification](../../docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
- [Phase 1-2 Completion](../../docs/PHASE_1_2_COMPLETION.md)
- [Work Log](../../docs/work-logs/2026-02-10_clip-integration.md)
