# LSTM-ViT Dynamic Gesture Recognition Workflow Plan

## Overview
This workflow integrates a Vision Transformer (ViT) for per-frame feature extraction with a Long Short-Term Memory (LSTM) network for temporal modeling of dynamic gestures. The LSTM processes sequences of ViT features to recognize and describe gesture evolution over time.

## Key Steps
1. **Frame Selection**: Sample a reasonable number of frames from input video or stream.
2. **ViT Feature Extraction**: Use the existing ViT implementation to extract features for each frame.
3. **LSTM Temporal Modeling**: Feed the sequence of ViT features into an LSTM to capture gesture dynamics.
4. **Gesture Description**: The LSTM output is mapped to a semantic description or class of the dynamic gesture.

## File Structure
- `src/recognition_engine/temporal/vit_lstm_pipeline.py`: Main pipeline for ViT-LSTM workflow.
- `src/recognition_engine/temporal/lstm_model.py`: LSTM model definition.
- `scripts/train_vit_lstm.py`: Training script for the pipeline.
- `scripts/infer_vit_lstm.py`: Inference/demo script.
- `docs/Recognition-Engine/LSTM_VIT_WORKFLOW.md`: Documentation and usage guide.

## Model Choice
- Use a Bi-directional LSTM with attention for best performance and real-time capability.
- PyTorch implementation for compatibility with existing codebase.

## Commit Plan
- [ ] Add initial plan and file structure.
- [ ] Implement feature extraction and LSTM modules.
- [ ] Add training/inference scripts.
- [ ] Document workflow and usage.
