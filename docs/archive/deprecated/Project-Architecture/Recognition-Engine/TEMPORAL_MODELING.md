# Temporal Modeling for Gesture Recognition

This document formalizes the interface between the ViT-based frame encoder and a pluggable temporal head (LSTM now; Mamba in the future).

## Interfaces and Shapes

- ViT Feature per Frame: D=768 for ViT-B/16 (after removing classification head)
- Batching and sequences:
  - Images: (B, T, 3, H=224, W=224)
  - Embeddings: (B, T, D)
- Normalization: Use `timm` default transforms for ViT-B/16 (mean/std normalization). Embeddings may be L2-normalized or projected if needed, but the temporal input dimension remains D.

## Modules

- ViT Feature Extractor: Frozen by default; produces one embedding per frame.
- Temporal Head: Consumes (B, T, D) and outputs a dict:
  - `static_logits`: (B, T, C_static) per-frame static gesture logits (optional)
  - `dynamic_logits`: (B, C_dynamic) sequence-level dynamic gesture logits (optional)

See `src/recognition_engine/temporal/temporal_heads.py`:

- `TemporalHeadBase`: Interface definition
- `LSTMTemporalHead`: Two-layer (bi)LSTM implementation
- `TemporalClassifier`: Wrapper combining ViT features and a temporal head

## Responsibilities

- ViT (frame-level): Encodes instantaneous hand-shape evidence from each ROI frame.
- Temporal head (sequence-level): Aggregates embeddings over time to model persistence and transitions, classifying static vs. dynamic gestures.

## Static vs. Dynamic Handling

- Static: Use per-frame logits with temporal smoothing (e.g., majority or EMA). Optionally, treat the sequence as one class if labels are per-clip.
- Dynamic: Use sequence logits based on pooled hidden states (mean over time or last state).

## Extensibility (Mamba)

- Requirements to swap LSTM -> Mamba without upstream changes:
  - Keep input `(B, T, D)` unchanged
  - Preserve output dict keys/shapes
  - No changes to ViT weights or dataset preprocessing
- Implement `MambaTemporalHead` with identical interface (`TemporalHeadBase`).

## Practical Configs

- Typical window T=32–64 frames (overlap 50% for streaming)
- LSTM hidden=256–512, layers=2, bidirectional=True, dropout=0.2
- Use gradient clipping and label smoothing for stability

## Notes

- The phrase “768×768 feature maps” is corrected to “768-d embeddings per frame”. If multi-layer ViT features are needed, concatenate or project to a fixed D before the temporal head.
