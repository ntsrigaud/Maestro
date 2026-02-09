# Quick Reference - ViT Training for Gesture Recognition

## Essential Commands

### Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate gesture

# Or manual setup
conda create -n gesture python=3.10 -y
conda activate gesture
conda install -c pytorch pytorch torchvision -y
conda install -c conda-forge timm pillow matplotlib seaborn pandas tqdm scikit-learn jupyter -y
```

### Download Dataset (FAST METHOD - 5-15 minutes)

```bash
# ⭐ RECOMMENDED: Kaggle pre-made subset (1.5 GB, ~30K images)
pip install kaggle
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset

# Filter to specific gestures (optional, reduces to ~600 MB)
python scripts/download_hagrid_kaggle.py \
    --gestures fist palm peace like dislike stop ok call \
    --samples_per_class 400

# ⚠️  OLD METHOD (SLOW - downloads 280+ GB, DO NOT USE):
# python scripts/download_hagrid_subset.py  # <-- Too slow!
```

**Note**: See `docs/HAGRID_DOWNLOAD_SOLUTIONS.md` for all options and size comparison.

### Verify Dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('data/hagrid_subset/annotations.csv'); print(df['label'].value_counts())"
```

### Launch Training

```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

### Check Disk Usage

```bash
du -sh data/hagrid_subset
du -sh checkpoints
```

## Training in Notebook

### Key Cells to Modify

**Cell 5 - Dataset Path**

```python
DATASET_ROOT = './data/hagrid_subset'  # Update to your path
```

**Cell 7 - Model Selection**

```python
MODEL_NAME = 'vit_base_patch16_224'  # Try: deit_small_patch16_224, swin_small_patch4_window7_224
```

**Cell 9 - Hyperparameters**

```python
LEARNING_RATE = 3e-5      # Try: 1e-5, 5e-5, 1e-4
EPOCHS = 10               # Try: 15, 20, 30
BATCH_SIZE = 32           # Reduce if OOM: 16, 8
```

## Expected Training Output

```
Epoch 1/10
------------------------------------------------------------
Train Loss: 1.2543 | Acc: 0.6234 | F1: 0.6102
Val   Loss: 0.8765 | Acc: 0.7543 | F1: 0.7421
Time: 123.4s | LR: 3.00e-06
✅ Saved checkpoint: ../checkpoints/vit_hagrid_best.pt (val_acc=0.7543)

Epoch 2/10
------------------------------------------------------------
Train Loss: 0.6321 | Acc: 0.8012 | F1: 0.7943
Val   Loss: 0.5432 | Acc: 0.8321 | F1: 0.8234
Time: 119.2s | LR: 1.50e-05
✅ Saved checkpoint: ../checkpoints/vit_hagrid_best.pt (val_acc=0.8321)

...

🎉 Training complete!
Best validation accuracy: 0.8921
Best validation F1: 0.8834
```

## Model Export (After Training)

### TorchScript

```python
model.eval()
example_input = torch.randn(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'models/vit_hagrid_gesture.pt')
```

### ONNX

```python
torch.onnx.export(
    model, example_input, 'models/vit_hagrid_gesture.onnx',
    input_names=['image'], output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}
)
```

## Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms

# Load checkpoint
checkpoint = torch.load('checkpoints/vit_hagrid_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
image = Image.open('hand.jpg')
input_tensor = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(input_tensor)
    pred = output.argmax(dim=1).item()
    conf = torch.softmax(output, dim=1)[0][pred].item()

gesture = checkpoint['class_names'][pred]
print(f'Gesture: {gesture} (confidence: {conf:.2%})')
```

## Performance Metrics

### Target Metrics (10 classes, 800 samples/class)

- **Test Accuracy**: 85-90%
- **Test F1 Score**: 83-88%
- **Inference Time (GPU)**: 10-30ms
- **Inference Time (CPU)**: 50-150ms
- **Model Size**: ~330 MB

### Typical Results by Model

| Model       | Accuracy | Inference (GPU) | Size  |
| ----------- | -------- | --------------- | ----- |
| ViT-B/16    | 88-92%   | 15-25ms         | 330MB |
| DeiT-Small  | 85-89%   | 10-15ms         | 85MB  |
| Swin-Small  | 87-91%   | 20-30ms         | 190MB |
| MobileViT-S | 82-86%   | 8-12ms          | 22MB  |

## Troubleshooting Quick Fixes

### Out of Memory

```python
BATCH_SIZE = 8  # Reduce from 32
NUM_WORKERS = 2  # Reduce from 4
```

### Slow Training

```bash
# Check device
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

### Dataset Not Loading

```bash
# Verify CSV exists
ls -lh data/hagrid_subset/annotations.csv

# Check first few rows
head -n 5 data/hagrid_subset/annotations.csv
```

### Low Accuracy

- Increase `samples_per_class` to 1000+
- Train longer (20-30 epochs)
- Try different model (Swin)
- Add more augmentation

## Directory Structure Checklist

```
✓ environment.yml
✓ TRAINING_README.md
✓ notebooks/finetune_vit_hagrid.ipynb
✓ scripts/download_hagrid_subset.py
✓ docs/HaGRID_DATASET_ASSESSMENT.md
✓ docs/HAGRID_DOWNLOAD_GUIDE.md
✓ docs/TRAINING_SETUP_SUMMARY.md
□ data/hagrid_subset/               # Created by download script
  □ annotations.csv
  □ train/, val/, test/
□ checkpoints/                       # Created during training
  □ vit_hagrid_best.pt
□ models/                            # Created during export
  □ vit_hagrid_gesture.pt (TorchScript)
  □ vit_hagrid_gesture.onnx
```

## Resources

- 📂 **[Full Training Guide](TRAINING_README.md)**
- 📊 **[Dataset Assessment](docs/HaGRID_DATASET_ASSESSMENT.md)**
- 📥 **[Download Guide](docs/HAGRID_DOWNLOAD_GUIDE.md)**
- 📝 **[Setup Summary](docs/TRAINING_SETUP_SUMMARY.md)**
- 🔗 **[HaGRID GitHub](https://github.com/hukenovs/hagrid)**
- 📄 **[HaGRID Paper](https://arxiv.org/abs/2412.01508)**

## Common Workflows

### First-Time Setup (Fresh Start)

```bash
conda env create -f environment.yml
conda activate gesture
python scripts/download_hagrid_subset.py --download_annotations --save_path ./data/hagrid_subset
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
# Run all cells
```

### Resume Training

```bash
conda activate gesture
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
# Modify checkpoint loading to resume from epoch N
```

### Try Different Model

```bash
# In notebook, cell 7:
MODEL_NAME = 'deit_small_patch16_224'
# Rerun cells 7-12
```

### Expand Dataset

```bash
python scripts/download_hagrid_subset.py \
    --save_path ./data/hagrid_subset \
    --gestures call dislike fist like ok palm peace point rock no_gesture four mute one \
    --samples_per_class 1000
# Update notebook DATASET_ROOT and rerun
```

## Status Checks

### Dataset Ready?

```bash
[ -f data/hagrid_subset/annotations.csv ] && echo "✅ Dataset ready" || echo "❌ Run download script"
```

### Environment Ready?

```bash
conda env list | grep gesture && echo "✅ Environment exists" || echo "❌ Create environment"
```

### Training Complete?

```bash
[ -f checkpoints/vit_hagrid_best.pt ] && echo "✅ Model trained" || echo "❌ Train model"
```

## Time Estimates

| Task                        | Time (GPU) | Time (CPU) |
| --------------------------- | ---------- | ---------- |
| Download subset (800/class) | 2-4 hours  | 2-4 hours  |
| First training (10 epochs)  | 30-60 min  | 2-3 hours  |
| Evaluation                  | 2-5 min    | 5-10 min   |
| Export (TorchScript + ONNX) | <1 min     | <2 min     |

## Contact / Issues

For issues specific to this training setup, check:

- `docs/TRAINING_SETUP_SUMMARY.md` - Complete overview
- `docs/HAGRID_DOWNLOAD_GUIDE.md` - Download troubleshooting

For HaGRID dataset issues:

- https://github.com/hukenovs/hagrid/issues
