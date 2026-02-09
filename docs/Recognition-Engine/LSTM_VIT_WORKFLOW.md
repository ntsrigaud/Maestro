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

## Usage

### Training

Run the training script with your dataset organized by gesture class and sequence:

```bash
python scripts/train_vit_lstm.py --data_dir path/to/data --epochs 20 --batch_size 8 --device cuda
```

### Inference

Run inference on a folder of frames:

```bash
python scripts/infer_vit_lstm.py --frames_dir path/to/frames --lstm_weights lstm_vit_best.pth --device cuda
```

### Data Format

- Training data should be organized as:
    - `data_dir/gesture_class/sequence_id/frame_001.jpg ...`
- Inference expects a folder of frames for a single gesture instance.

### Model Details

- ViT extracts per-frame features (default: 768-dim for base model).
- BiLSTM with attention models temporal evolution.
- Output is a gesture class and a semantic description.

### Customization

- Update `gesture_labels` in scripts for your dataset.
- Adjust LSTM parameters in scripts for your use case.

### Real-Time Use

- For real-time, sample frames from a video stream, extract features, and run through the pipeline as shown in the inference script.
