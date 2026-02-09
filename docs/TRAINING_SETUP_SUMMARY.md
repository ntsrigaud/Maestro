# HaGRID Dataset Assessment & Training Setup - Summary

**Date**: December 4, 2025  
**Branch**: `finetune-vit-hagrid`

## Completed Tasks

### 1. HaGRID Dataset Assessment

**Findings:**

- HaGRID v2-1M is a massive dataset (1.5TB, 1M+ images, 33 gesture classes)
- Per-class archives range from 37-54 GB each
- Dataset is split into separate ZIPs by gesture class (enables selective download)
- Annotations available in JSON (COCO format) with bounding boxes, landmarks, and metadata

**Solution for Storage Constraints:**

- Created custom subset downloader that samples N images per class
- Maintains train/val/test split ratios (76%/9%/15%)
- Target: 10 gesture classes × 800 samples = ~2.5 GB total

**Selected Gestures (from screenshot):**

1. call
2. dislike
3. fist
4. like
5. ok
6. palm
7. peace
8. point
9. rock
10. no_gesture

**Documentation:**

- `docs/HaGRID_DATASET_ASSESSMENT.md` - Detailed assessment report
- `docs/HAGRID_DOWNLOAD_GUIDE.md` - Download instructions and troubleshooting

### 2. Dataset Download Script

**Created**: `scripts/download_hagrid_subset.py`

**Features:**

- Selective gesture class downloading
- Random stratified sampling (preserves split ratios)
- Progress tracking with tqdm
- Generates simplified CSV annotations (`image_path`, `label`, `split`)
- Folder-per-class structure for PyTorch compatibility

**Usage:**

```bash
python scripts/download_hagrid_subset.py \
    --save_path ./data/hagrid_subset \
    --gestures call dislike fist like ok palm peace point rock no_gesture \
    --samples_per_class 800 \
    --download_annotations
```

### 3. Training Notebook

**Created**: `notebooks/finetune_vit_hagrid.ipynb`

**Contents:**

1. **Environment Setup** - Conda instructions for PyTorch, timm, etc.
2. **Dataset Loading** - Custom PyTorch Dataset for HaGRID subset CSV
3. **Data Augmentation** - Transforms (resize, flip, rotation, color jitter)
4. **Visualization** - Sample images from each class
5. **Model Selection** - ViT-B/16 (default), with notes on alternatives (DeiT, Swin, MobileViT)
6. **Training Configuration** - Hyperparameters, AdamW optimizer, cosine annealing scheduler
7. **Training Loop** - Epoch-based training with progress bars, metrics logging
8. **Evaluation** - Test set evaluation with classification report and confusion matrix
9. **Training History Plots** - Loss, accuracy, F1 curves
10. **Next Steps** - Model export (TorchScript/ONNX), integration, optimization

**Key Features:**

- GPU/MPS/CPU auto-detection
- Best model checkpointing (saves to `../checkpoints/vit_hagrid_best.pt`)
- F1 score and accuracy tracking
- Confusion matrix visualization
- Ready for immediate use after dataset download

### 4. Model Architecture Selection

**Primary Model**: `vit_base_patch16_224`

- 86M parameters
- ImageNet pretrained
- 224×224 input size
- Good baseline for fine-tuning

**Alternative Candidates** (documented in notebook):

- `deit_small_patch16_224` - More efficient (22M params)
- `swin_small_patch4_window7_224` - Hierarchical, better for details (50M params)
- `mobilevit_s` - Lightweight for real-time (5.6M params)

### 5. Preprocessing Pipeline

**Implemented:**

- Resize to 224×224 (ViT input size)
- Data augmentation:
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)
- Separate train/val transforms

**Designed for Integration:**

- Accepts cropped hand images (output from hand detection module)
- Same preprocessing can be applied at inference time

## File Structure

```
CSU-Capstone-Project/
├── notebooks/
│   └── finetune_vit_hagrid.ipynb          # Main training notebook
├── scripts/
│   └── download_hagrid_subset.py          # Dataset downloader
├── docs/
│   ├── HaGRID_DATASET_ASSESSMENT.md       # Dataset analysis
│   └── HAGRID_DOWNLOAD_GUIDE.md           # Download instructions
├── data/
│   └── hagrid_subset/                     # (To be created by user)
│       ├── annotations.csv
│       ├── train/
│       ├── val/
│       └── test/
└── checkpoints/                            # (Created during training)
    └── vit_hagrid_best.pt
```

## Next Steps (User Actions Required)

### 1. Download Dataset (~2-4 hours, depending on network)

```bash
# Activate conda environment
conda create -n gesture python=3.10 -y
conda activate gesture

# Install dependencies
conda install -c pytorch pytorch torchvision -y
conda install -c conda-forge timm pillow -y
pip install pandas tqdm matplotlib seaborn scikit-learn

# Download subset
python scripts/download_hagrid_subset.py \
    --save_path ./data/hagrid_subset \
    --gestures call dislike fist like ok palm peace point rock no_gesture \
    --samples_per_class 800 \
    --download_annotations
```

### 2. Verify Dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('data/hagrid_subset/annotations.csv'); print(df['label'].value_counts())"
```

Expected output: ~800 images per class across 10 classes

### 3. Open Training Notebook

```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

### 4. Update Configuration in Notebook

In cell 5, update:

```python
DATASET_ROOT = './data/hagrid_subset'  # Or your actual path
```

### 5. Run Training

Execute all cells sequentially:

1. Environment check
2. Dataset loading
3. Visualization (verify data)
4. Model creation
5. Training configuration
6. Training loop (10 epochs, ~30-60 min on GPU)
7. Plot training curves
8. Evaluate on test set

### 6. Monitor Training

Expected results (after 10 epochs on 8K images):

- **Training accuracy**: 95-98%
- **Validation accuracy**: 85-92%
- **Test accuracy**: 85-90%
- **Checkpoint**: Saved to `checkpoints/vit_hagrid_best.pt`

## Model Alternatives (Future Experimentation)

Try different models by changing `MODEL_NAME` in the notebook:

| Model                           | Params | Speed     | Best For                |
| ------------------------------- | ------ | --------- | ----------------------- |
| `vit_base_patch16_224`          | 86M    | Medium    | Baseline, high accuracy |
| `deit_small_patch16_224`        | 22M    | Fast      | Balanced speed/accuracy |
| `swin_small_patch4_window7_224` | 50M    | Medium    | Fine-grained gestures   |
| `mobilevit_s`                   | 5.6M   | Very Fast | Real-time inference     |

## Performance Targets

- **Accuracy**: >85% on test set (10 classes)
- **Inference Latency**: <50ms per image (GPU), <200ms (CPU)
- **Model Size**: <350MB (ViT-B/16 checkpoint)

## Integration Plan

After training completes:

1. **Export Model**:
   - TorchScript (`.pt`) for PyTorch inference
   - ONNX (`.onnx`) for cross-platform deployment

2. **Create Inference Script**:
   - `src/recognition_engine/gesture_classifier.py`
   - Input: Cropped hand image (from hand detection)
   - Output: Gesture class + confidence

3. **Connect to Hand Detection**:
   - Hand detection module outputs bounding boxes
   - Crop images using bboxes
   - Pass crops to gesture classifier
   - Return gesture predictions

4. **Real-time Testing**:
   - Test on live webcam feed
   - Measure end-to-end latency
   - Optimize if needed (quantization, model size reduction)

## References

- **HaGRID Repository**: https://github.com/hukenovs/hagrid
- **HaGRID Paper**: https://arxiv.org/abs/2412.01508
- **timm Library**: https://github.com/huggingface/pytorch-image-models
- **Vision Transformers**: https://arxiv.org/abs/2010.11929

## Troubleshooting

### Dataset Download Issues

- **Problem**: Download fails or is very slow
- **Solution**: Use wired connection, run overnight, or reduce `samples_per_class`

### Out of Memory During Training

- **Problem**: CUDA/MPS out of memory
- **Solution**: Reduce `BATCH_SIZE` (try 16 or 8), or use a smaller model

### Low Accuracy

- **Problem**: Test accuracy <80%
- **Solution**: Increase `samples_per_class`, add more augmentation, train longer (20-30 epochs)

### Slow Inference

- **Problem**: >100ms per image
- **Solution**: Switch to `mobilevit_s` or `deit_small`, use quantization, batch inference

## Summary

✅ **Assessment Complete**: HaGRID dataset analyzed, subset strategy defined  
✅ **Downloader Ready**: Script to fetch 2.5GB subset (~8K images, 10 classes)  
✅ **Training Pipeline**: Full notebook with ViT fine-tuning, evaluation, visualization  
✅ **Documentation**: Comprehensive guides for dataset and training

**User is ready to**:

1. Download subset
2. Train ViT model
3. Achieve 85-90% gesture recognition accuracy
4. Export model for integration with hand detection module
