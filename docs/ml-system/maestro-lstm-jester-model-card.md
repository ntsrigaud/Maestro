---
language: en
license: mit
library_name: pytorch
tags:
  - gesture-recognition
  - human-computer-interaction
  - mediapipe
  - lstm
  - transfer-learning
pipeline_tag: image-classification
---

# maestro-lstm-jester

Model card for Hugging Face repository: `ntsrigaud/maestro-lstm-jester`

## Model Details

- Architecture: `EnhancedTwoStreamLSTM` (`two_stream_attn_v1`)
- Approx parameter count: ~1.20M
- Backbone:
  - Two-stream BiLSTM encoder
  - Multi-head self-attention per stream (8 heads)
  - Cross-stream MLP gating
- Classification head:
  - `LayerNorm(384) -> Linear(384,192) -> GELU -> Dropout -> Linear(192,C)`
- Input shape: `(batch, 32, 147)`
- Feature schema: `feature-schema-v5`
- Class counts:
  - Pretraining: 27 (Jester)
  - Fine-tuning: 14 (IPN)

## Training Data

### Phase 1: Jester pretraining

- Dataset: 20BN-Jester-v1
- Split used for training pipeline:
  - train: 0.85
  - val: 0.10
  - test: 0.05
- Labels: 27 gesture classes
- Input source: per-video JPEG sequences with MediaPipe Holistic extraction

### Phase 2: IPN fine-tuning

- Dataset: IPN Hand
- Split used for training pipeline:
  - train: 0.70
  - val: 0.15
  - test: 0.15
- Labels: full 14-class IPN vocabulary
- Input source: right-hand landmarks transformed to `feature-schema-v5`

## Training Procedure

### Phase 1 config (Jester)

- Config file: `configs/model/two_stream_attn_jester.yaml`
- Epochs: 80
- Batch size: 256
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `5e-4`
- Label smoothing: `0.05`
- Class weighting: enabled

### Phase 2 config (IPN)

- Config file: `configs/model/two_stream_attn_ipn_finetune.yaml`
- Stage A (head-only warm-up):
  - frozen backbone, new head only
  - epochs: 10
  - learning rate: `1e-3`
- Stage B (full fine-tune):
  - unfreeze all layers
  - epochs: 60
  - learning rate: `3e-5`
- Batch size: 128
- Weight decay: `5e-4`
- Label smoothing: `0.05`

## Intended Use

- Real-time hand-gesture control driven by MediaPipe landmarks
- Research baseline for transfer learning from Jester to IPN
- Evaluation and comparison with prior IPN baselines

## Limitations

- Depends on landmark quality; poor hand tracking degrades predictions
- Domain shift across camera viewpoints, lighting, and users can reduce performance
- Not a safety-certified model for critical decision systems

## Evaluation

Use the output of `notebooks/07_model_training_jester_pretrain_ipn_finetune.ipynb`
for exact final metrics (test accuracy, macro-F1, per-class recall, confusion matrix).

Suggested reporting table:

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| C3D-LSTM (IPN baseline) | 76.4 | - |
| I3D (IPN baseline) | 79.9 | - |
| TwoStream LSTM v1 (internal) | 86.8 | - |
| EnhancedTwoStreamLSTM (Jester->IPN) | fill_after_run | fill_after_run |

## Reproducibility

1. Build datasets in `notebooks/04_dataset_preparation_jester_ipn.ipynb`.
2. Run two-phase training in `notebooks/07_model_training_jester_pretrain_ipn_finetune.ipynb`.
3. Export artifact using `export_inference_artifact`.
4. Publish with `publish_to_huggingface(..., repo_id="ntsrigaud/maestro-lstm-jester")`.

## Citation

If you use this model, cite:

- 20BN-Jester-v1 dataset
- IPN Hand dataset
- This Maestro repository and model card
