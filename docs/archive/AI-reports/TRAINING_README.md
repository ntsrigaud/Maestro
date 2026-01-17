# ViT Fine-tuning for Hand Gesture Recognition

This directory contains all materials for fine-tuning a Vision Transformer (ViT) on the HaGRID dataset for real-time hand gesture recognition.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment from YAML
conda env create -f environment.yml
conda activate gesture

# Or create manually
conda create -n gesture python=3.10 -y
conda activate gesture
conda install -c pytorch pytorch torchvision -y
conda install -c conda-forge timm pillow matplotlib seaborn pandas tqdm scikit-learn jupyter -y
pip install transformers datasets accelerate
```

### 2. Download HaGRID Subset

** RECOMMENDED METHOD** - Download pre-made Kaggle subset (~1.5GB, 5-15 minutes):

```bash
pip install kaggle
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset

# Optional: Filter to specific gestures (reduces to ~600 MB)
python scripts/download_hagrid_kaggle.py \
    --gestures call dislike fist like ok palm peace point rock \
    --samples_per_class 400
```

** OLD METHOD (NOT RECOMMENDED)**: The original script (`download_hagrid_subset.py`) downloads 280+ GB of full dataset ZIPs before extracting subsets. This is extremely slow and requires massive temporary storage. See `docs/HAGRID_DOWNLOAD_SOLUTIONS.md` for details.

### 3. Verify Dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('data/hagrid_subset/annotations.csv'); print(f'Total images: {len(df)}'); print(df['label'].value_counts())"
```

Expected: ~8,000 images across 10 gesture classes

### 4. Train ViT Model

Open and run the training notebook:

```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

**Training time**: ~30-60 minutes on GPU (M1/M2 Mac or NVIDIA), 2-3 hours on CPU

### 5. Expected Results

After training (10 epochs):

- **Test Accuracy**: 85-90%
- **Model Checkpoint**: `checkpoints/vit_hagrid_best.pt`
- **Model Size**: ~330 MB

## Documentation

- **[HaGRID Dataset Assessment](docs/HaGRID_DATASET_ASSESSMENT.md)** - Detailed analysis of the dataset
- **[Download Guide](docs/HAGRID_DOWNLOAD_GUIDE.md)** - Instructions and troubleshooting for dataset download
- **[Training Setup Summary](docs/TRAINING_SETUP_SUMMARY.md)** - Complete walkthrough of what was built

## Project Structure

```
.
├── environment.yml                        # Conda environment specification
├── notebooks/
│   └── finetune_vit_hagrid.ipynb         # Training notebook (main workflow)
├── scripts/
│   └── download_hagrid_subset.py         # Dataset downloader
├── docs/
│   ├── HaGRID_DATASET_ASSESSMENT.md      # Dataset analysis
│   ├── HAGRID_DOWNLOAD_GUIDE.md          # Download instructions
│   └── TRAINING_SETUP_SUMMARY.md         # Setup summary
├── data/
│   └── hagrid_subset/                    # Dataset (created by download script)
│       ├── annotations.csv               # Image paths and labels
│       ├── train/                        # Training images
│       ├── val/                          # Validation images
│       └── test/                         # Test images
└── checkpoints/                           # Model checkpoints (created during training)
    └── vit_hagrid_best.pt
```

## Gesture Classes

The selected 10 gesture classes (from HaGRID):

| Gesture        | Description                              |
| -------------- | ---------------------------------------- |
| **call**       | Phone gesture (thumb and pinky extended) |
| **dislike**    | Thumbs down                              |
| **fist**       | Closed fist                              |
| **like**       | Thumbs up                                |
| **ok**         | OK sign (thumb and index touching)       |
| **palm**       | Open palm                                |
| **peace**      | Peace sign (V gesture)                   |
| **point**      | Pointing gesture                         |
| **rock**       | Rock-on sign                             |
| **no_gesture** | Background/no gesture                    |

## Model Options

The notebook supports multiple model architectures. Change `MODEL_NAME` in the notebook:

| Model                           | Parameters | Speed     | Best For                    |
| ------------------------------- | ---------- | --------- | --------------------------- |
| `vit_base_patch16_224`          | 86M        | Medium    | **Default - High accuracy** |
| `deit_small_patch16_224`        | 22M        | Fast      | Balanced efficiency         |
| `swin_small_patch4_window7_224` | 50M        | Medium    | Fine-grained details        |
| `mobilevit_s`                   | 5.6M       | Very Fast | Real-time inference         |

## Training Configuration

Default hyperparameters (adjustable in notebook):

- **Learning Rate**: 3e-5
- **Weight Decay**: 0.01
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: AdamW with cosine annealing
- **Warmup**: 1 epoch
- **Image Size**: 224×224
- **Augmentation**: Horizontal flip, rotation (±15°), color jitter

## Next Steps After Training

### 1. Export Model

```python
# In the notebook or a script
model.eval()
example_input = torch.randn(1, 3, 224, 224).to(device)

# TorchScript export
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'models/vit_hagrid_gesture.pt')

# ONNX export
torch.onnx.export(model, example_input, 'models/vit_hagrid_gesture.onnx')
```

### 2. Create Inference Script

Integrate the trained model with the hand detection module:

```python
# Example inference
from PIL import Image
import torch
from torchvision import transforms

# Load model
checkpoint = torch.load('checkpoints/vit_hagrid_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess hand crop
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference
hand_crop = Image.open('hand_image.jpg')
input_tensor = transform(hand_crop).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

print(f'Gesture: {class_names[predicted_class]}, Confidence: {confidence:.2f}')
```

### 3. Performance Optimization

- **Quantization**: Reduce model size and improve inference speed
- **Pruning**: Remove redundant weights
- **Batch Inference**: Process multiple hands simultaneously
- **Model Distillation**: Train a smaller student model

## Troubleshooting

### Download Issues

**Problem**: Download fails or is very slow  
**Solution**:

- Use a wired connection
- Run the download script overnight
- Reduce `--samples_per_class` to 500

### Out of Memory

**Problem**: CUDA/MPS out of memory during training  
**Solution**:

- Reduce `BATCH_SIZE` to 16 or 8
- Use a smaller model (`deit_small_patch16_224`)
- Close other applications

### Low Accuracy

**Problem**: Test accuracy below 80%  
**Solution**:

- Increase `--samples_per_class` to 1000+
- Train for more epochs (20-30)
- Add more data augmentation
- Try a different model architecture

### Slow Inference

**Problem**: >100ms per image  
**Solution**:

- Switch to `mobilevit_s`
- Use quantization
- Enable batch inference
- Use GPU/MPS if available

## References

- **HaGRID Repository**: https://github.com/hukenovs/hagrid
- **HaGRID Paper**: [arXiv:2412.01508](https://arxiv.org/abs/2412.01508)
- **Vision Transformers**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **timm Library**: https://github.com/huggingface/pytorch-image-models

## License

- **Code**: MIT (this project)
- **HaGRID Dataset**: Creative Commons Attribution-ShareAlike 4.0 International

## Contributing

To add more gestures or improve the pipeline:

1. Add gesture names to `--gestures` argument
2. Adjust `samples_per_class` based on available storage
3. Re-run the download script
4. Retrain the model

## Tips

- **Start small**: Train on 5-6 core gestures first, then expand
- **Monitor training**: Watch for overfitting (train acc >> val acc)
- **Test variations**: Try different learning rates (1e-5, 5e-5)
- **Use GPU**: Training is 10-20× faster on GPU/MPS
- **Save checkpoints**: Enable frequent checkpointing for long runs

## Learning Resources

- [Vision Transformers Explained](https://arxiv.org/abs/2010.11929)
- [Fine-tuning Pretrained Models](https://huggingface.co/docs/transformers/training)
- [Data Augmentation Best Practices](https://arxiv.org/abs/1906.11172)
