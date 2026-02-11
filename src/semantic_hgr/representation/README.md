# Representation Layer (Layer 2)

**Purpose**: Extract semantically aligned visual features from gesture images using CLIP.

## Overview

This layer transforms raw hand crop images from Layer 1 (Perception) into 512-dimensional embeddings that reside in CLIP's joint vision-language space. These embeddings enable semantic understanding and zero-shot gesture recognition.

## Components

### CLIPEncoder

**File**: `clip_encoder.py`

**Responsibility**: Encapsulate CLIP ViT-B/32 model for image encoding

**Architecture**:

- Input: NumPy RGB images (H, W, 3)
- Output: L2-normalized 512-dim embeddings
- Model: OpenAI CLIP ViT-B/32 pretrained weights

**Key Features**:

- Single image encoding: `encode_image()`
- Batch encoding: `encode_batch()`
- Deterministic behavior
- Clean NumPy interface (no tensor leakage)

**Usage Example**:

```python
from representation import CLIPEncoder
import numpy as np

# Initialize encoder
encoder = CLIPEncoder(model_name='ViT-B-32', device='cpu')

# Encode hand crop image
hand_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
embedding = encoder.encode_image(hand_image)

# Result: (512,) float32 L2-normalized embedding
print(embedding.shape)  # (512,)
print(np.linalg.norm(embedding))  # ~1.0
```

## Architecture Compliance

✅ **Clean Boundaries**: No dependency on Layer 3 (Semantic Intent)  
✅ **Interface Contract**: NumPy arrays in/out, no PyTorch tensors  
✅ **Error Handling**: Explicit validation with actionable messages  
✅ **Testability**: Fully covered by unit tests

## Performance

- **Encoding time**: ~30ms (ViT-B/32 on CPU)
- **Embedding dimension**: 512
- **Memory footprint**: ~605MB (model weights)

## Testing

Run Layer 2 tests:

```bash
python -m unittest tests.test_clip_encoder -v
```

**Coverage**: 22 tests validating initialization, encoding, error handling, edge cases

## Future Enhancements

1. **GPU Acceleration**: CUDA/MPS support for faster encoding
2. **Model Upgrade**: ViT-L/14 for better accuracy (768-dim embeddings)
3. **Temporal Aggregation**: Extend to video sequences (Phase 3)

## Dependencies

- `torch>=2.0.0`
- `open-clip-torch>=2.20.0`
- `pillow>=9.0.0`
- `numpy>=1.24.0`

## See Also

- [Architecture Specification](../../docs/SEMANTIC_SYSTEM_ARCHITECTURE.md)
- [Phase 1-2 Completion](../../docs/PHASE_1_2_COMPLETION.md)
- [Work Log](../../docs/work-logs/2026-02-10_clip-integration.md)
