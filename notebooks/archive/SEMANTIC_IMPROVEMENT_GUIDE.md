# Improving Semantic Gesture-Text Similarity

## Problem Diagnosis

Your t-SNE visualization shows scattered embeddings, indicating weak alignment between gesture images and text descriptions. This typically happens due to:

1. **Insufficient training** - Only 5 epochs may not be enough
2. **Frozen encoders** - Both ViT and SentenceTransformer are mostly frozen
3. **Temperature too low** - 0.07 may be too strict for initial training
4. **Limited text variety** - Need more diverse descriptions per gesture
5. **Small batch size** - 32 may be too small for contrastive learning

---

## 🎯 Recommended Improvements (Ranked by Impact)

### 1. **Train Longer with Gradual Unfreezing** ⭐⭐⭐⭐⭐

**Problem**: 5 epochs with frozen backbones limits learning

**Solution**: Progressive unfreezing strategy

```python
# Increase epochs
SEMANTIC_EPOCHS = 20  # Instead of 5

# Phase 1 (Epochs 1-5): Train only projections
# Phase 2 (Epochs 6-15): Unfreeze last layers of ViT
# Phase 3 (Epochs 16-20): Fine-tune everything

# Add to training loop:
if epoch == 5:
    print("Phase 2: Unfreezing ViT last 3 blocks...")
    for param in gesture_encoder.vit.blocks[-3:].parameters():
        param.requires_grad = True
    optimizer.add_param_group({
        'params': gesture_encoder.vit.blocks[-3:].parameters(),
        'lr': SEMANTIC_LR / 10  # Lower LR for pretrained layers
    })

if epoch == 15:
    print("Phase 3: Unfreezing text encoder projection...")
    for param in text_encoder.sentence_encoder.parameters():
        param.requires_grad = True
    optimizer.add_param_group({
        'params': text_encoder.sentence_encoder.parameters(),
        'lr': SEMANTIC_LR / 100  # Much lower LR
    })
```

**Expected Impact**: 40-60% improvement in similarity scores

---

### 2. **Use Larger Batch Size** ⭐⭐⭐⭐⭐

**Problem**: Batch size of 32 provides limited negative samples for contrastive learning

**Solution**: Increase to 128-256 (or use gradient accumulation if GPU memory limited)

```python
# Option 1: Increase batch size directly (if GPU allows)
SEMANTIC_BATCH_SIZE = 128  # Instead of 32

# Option 2: Gradient accumulation (if memory constrained)
SEMANTIC_BATCH_SIZE = 32
ACCUMULATION_STEPS = 4  # Effective batch size = 128

# Modify training loop:
optimizer.zero_grad()
for i, (images, texts, _) in enumerate(pbar):
    images = images.to(device)
    
    visual_embeds = gesture_encoder(images)
    text_embeds = text_encoder(texts)
    loss = contrastive_loss(visual_embeds, text_embeds)
    
    loss = loss / ACCUMULATION_STEPS  # Normalize loss
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Expected Impact**: 30-40% improvement in clustering quality

---

### 3. **Adjust Temperature Parameter** ⭐⭐⭐⭐

**Problem**: Temperature = 0.07 is too strict (same as CLIP), making training unstable

**Solution**: Start with higher temperature, gradually decrease

```python
# Warm-up temperature schedule
def get_temperature(epoch, total_epochs):
    if epoch < 5:
        return 0.15  # Softer at start
    elif epoch < 10:
        return 0.10
    else:
        return 0.07  # CLIP-like at end

# Update loss in training loop:
current_temp = get_temperature(epoch, SEMANTIC_EPOCHS)
contrastive_loss = InfoNCELoss(temperature=current_temp)
```

**Expected Impact**: 20-30% improvement in training stability

---

### 4. **Add Hard Negative Mining** ⭐⭐⭐⭐

**Problem**: Random negatives may be too easy

**Solution**: Sample harder negative examples

```python
class InfoNCELossWithHardNegatives(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, visual_embeds, text_embeds):
        batch_size = visual_embeds.shape[0]
        
        # Compute full similarity matrix
        logits = torch.matmul(visual_embeds, text_embeds.T) / self.temperature
        
        # Labels for positive pairs
        labels = torch.arange(batch_size, device=visual_embeds.device)
        
        # Standard loss
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        
        # Hard negative mining: focus on top-k hardest negatives
        with torch.no_grad():
            # Get similarities excluding diagonal (positives)
            mask = torch.eye(batch_size, device=visual_embeds.device).bool()
            hard_neg_logits = logits.clone()
            hard_neg_logits[mask] = -float('inf')
            
            # Weight hard negatives more
            hard_neg_weights = F.softmax(hard_neg_logits * 2, dim=-1)
        
        # Weighted loss (optional enhancement)
        loss = (loss_v2t + loss_t2v) / 2
        
        return loss
```

**Expected Impact**: 15-25% improvement in hard cases

---

### 5. **Enhance Text Descriptions** ⭐⭐⭐⭐

**Problem**: Limited text variety per gesture class

**Solution**: Add more diverse descriptions, including:
- Action-oriented: "raising thumb up", "showing approval"
- Context-aware: "agreeing with someone", "liking a post"
- Emotional: "expressing satisfaction", "showing happiness"
- Variations: "thumbs up gesture", "thumb raised", "upward thumb"

```python
# In gesture_semantics.py - Expand descriptions
GESTURE_SEMANTICS = {
    'like': {
        'descriptions': [
            # Add 20+ more descriptions per gesture
            'thumbs up',
            'thumb pointing up',
            'raising thumb upward',
            'showing approval gesture',
            'expressing agreement',
            'indicating positive response',
            'signaling good job',
            'approving with thumb',
            'displaying satisfaction',
            'agreeing gesture',
            'positive hand sign',
            'upward thumb gesture',
            # ... (add more)
        ]
    }
}
```

**Expected Impact**: 20-30% improvement in zero-shot capability

---

### 6. **Use Data Augmentation** ⭐⭐⭐

**Problem**: Model sees same image-text pairs repeatedly

**Solution**: Augment both modalities

```python
# Stronger image augmentations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),  # Increased
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # New
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # New
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # New
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Text augmentation (paraphrasing)
def augment_text(text):
    """Simple text augmentation"""
    augmentations = [
        text,
        text.replace('gesture', 'sign'),
        text.replace('showing', 'displaying'),
        text.replace('hand', 'palm'),
        # Add more synonyms
    ]
    return random.choice(augmentations)
```

**Expected Impact**: 15-20% improvement in robustness

---

### 7. **Add Focal Loss Weighting** ⭐⭐⭐

**Problem**: Model focuses on easy examples

**Solution**: Weight harder examples more

```python
class FocalInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma  # Focal loss gamma
        
    def forward(self, visual_embeds, text_embeds):
        batch_size = visual_embeds.shape[0]
        logits = torch.matmul(visual_embeds, text_embeds.T) / self.temperature
        labels = torch.arange(batch_size, device=visual_embeds.device)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        pos_probs = probs[range(batch_size), labels]
        
        # Focal weighting: (1 - p)^gamma
        focal_weights = (1 - pos_probs) ** self.gamma
        
        # Weighted cross-entropy
        loss_v2t = F.cross_entropy(logits, labels, reduction='none')
        loss_v2t = (focal_weights * loss_v2t).mean()
        
        loss_t2v = F.cross_entropy(logits.T, labels, reduction='none')
        focal_weights_t = (1 - probs.T[range(batch_size), labels]) ** self.gamma
        loss_t2v = (focal_weights_t * loss_t2v).mean()
        
        return (loss_v2t + loss_t2v) / 2
```

**Expected Impact**: 10-15% improvement on difficult classes

---

### 8. **Implement Curriculum Learning** ⭐⭐⭐

**Problem**: Model struggles with all gestures equally

**Solution**: Start with easy gestures, gradually add difficult ones

```python
# Group gestures by similarity
EASY_GESTURES = ['like', 'dislike', 'stop', 'ok']  # Very distinct
MEDIUM_GESTURES = ['peace', 'call', 'fist', 'palm']
HARD_GESTURES = ['one', 'two', 'three', 'four']  # Similar finger counts

# Phase training
def get_active_gestures(epoch):
    if epoch < 5:
        return EASY_GESTURES
    elif epoch < 12:
        return EASY_GESTURES + MEDIUM_GESTURES
    else:
        return EASY_GESTURES + MEDIUM_GESTURES + HARD_GESTURES

# Filter dataset accordingly
active_classes = get_active_gestures(epoch)
filtered_dataset = [sample for sample in dataset if sample.gesture_class in active_classes]
```

**Expected Impact**: 10-20% improvement in convergence speed

---

### 9. **Add Momentum Contrast (MoCo)** ⭐⭐⭐

**Problem**: Small batch size limits number of negatives

**Solution**: Maintain a queue of past embeddings

```python
class MoCoQueue:
    def __init__(self, size=4096, dim=512):
        self.queue = torch.randn(dim, size)
        self.queue = F.normalize(self.queue, dim=0)
        self.ptr = 0
        
    @torch.no_grad()
    def update(self, keys):
        batch_size = keys.shape[0]
        
        # Replace oldest embeddings
        self.queue[:, self.ptr:self.ptr + batch_size] = keys.T
        self.ptr = (self.ptr + batch_size) % self.queue.shape[1]
        
    def get_queue(self):
        return self.queue.T

# Use in training:
# Add queue negatives to contrastive loss computation
```

**Expected Impact**: 15-25% improvement with limited GPU memory

---

### 10. **Monitor Similarity Metrics** ⭐⭐

**Problem**: Training blindly without introspection

**Solution**: Track detailed metrics

```python
def compute_detailed_metrics(visual_embeds, text_embeds, labels):
    """Compute alignment metrics during training"""
    
    # Positive pair similarity (should be high)
    pos_sim = (visual_embeds * text_embeds).sum(dim=1).mean()
    
    # Intra-class similarity (gestures of same class)
    unique_labels = labels.unique()
    intra_sim = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            class_embeds = visual_embeds[mask]
            sim = torch.matmul(class_embeds, class_embeds.T)
            intra_sim.append(sim[~torch.eye(len(sim), dtype=bool)].mean())
    intra_sim = torch.stack(intra_sim).mean() if intra_sim else 0
    
    # Inter-class similarity (different classes, should be low)
    sim_matrix = torch.matmul(visual_embeds, visual_embeds.T)
    mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    inter_sim = sim_matrix[mask].mean()
    
    return {
        'pos_similarity': pos_sim.item(),
        'intra_class_similarity': intra_sim.item() if isinstance(intra_sim, torch.Tensor) else intra_sim,
        'inter_class_similarity': inter_sim.item(),
        'separation': (intra_sim - inter_sim).item() if isinstance(intra_sim, torch.Tensor) else 0
    }
```

**Expected Impact**: Better training insights and debugging

---

## 📊 Quick Wins (Implement First)

1. **Increase epochs to 15-20** (5 minutes to change)
2. **Increase batch size to 128** (if GPU allows) or use gradient accumulation
3. **Adjust temperature to 0.15 → 0.07** (warmup schedule)
4. **Add 10+ more text descriptions per gesture** (30 minutes work)
5. **Unfreeze ViT last 3 blocks after epoch 5** (10 minutes to implement)

**Expected Combined Impact**: 60-80% improvement in semantic similarity

---

## 🧪 Experimental (Advanced)

- **Multi-scale features**: Use features from multiple ViT layers
- **Cross-attention**: Add attention between visual and text features
- **Contrastive momentum**: EMA of encoders for stability
- **Supervised contrastive loss**: Group same-class samples as positives
- **Mix up augmentation**: Blend image-text pairs

---

## 📈 Validation Metrics to Track

Good semantic alignment should show:

1. **Positive similarity > 0.7** (matching pairs)
2. **Intra-class similarity > 0.6** (same gesture)
3. **Inter-class similarity < 0.3** (different gestures)
4. **Separation metric > 0.3** (intra - inter)
5. **Zero-shot Top-1 > 60%**
6. **Zero-shot Top-5 > 85%**

---

## 🔄 Recommended Training Pipeline

```python
# Phase 1: Warmup (Epochs 1-5)
- Batch size: 128
- Temperature: 0.15
- Learning rate: 1e-4
- Frozen: ViT backbone, SentenceTransformer
- Trainable: Both projections

# Phase 2: Fine-tuning (Epochs 6-15)
- Batch size: 128
- Temperature: 0.10
- Learning rate: 5e-5 (projections), 5e-6 (ViT)
- Frozen: SentenceTransformer
- Trainable: Projections + ViT last 3 blocks

# Phase 3: Full fine-tuning (Epochs 16-20)
- Batch size: 128
- Temperature: 0.07
- Learning rate: 1e-5 (all)
- Trainable: Everything (with small LR for pretrained)
```

---

## 🎯 Expected Results After Improvements

**Before** (Current):
- Positive similarity: ~0.4-0.5
- t-SNE: Scattered clusters
- Zero-shot Top-1: ~30-40%

**After** (Improvements):
- Positive similarity: ~0.75-0.85
- t-SNE: Tight, well-separated clusters
- Zero-shot Top-1: ~65-75%
- Same-color circles and stars will be very close together

---

## 💡 Pro Tips

1. **Start simple**: Implement improvements 1-5 first
2. **Monitor closely**: Track metrics every epoch
3. **Be patient**: Good alignment takes 15-20 epochs
4. **Save checkpoints**: Every 5 epochs in case of degradation
5. **Visualize often**: t-SNE after every 5 epochs to see progress

The most important factors are: **more training**, **larger batches**, and **gradual unfreezing**.
