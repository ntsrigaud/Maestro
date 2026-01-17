# Production Deployment Fixes - Implementation Guide

**Quick reference for implementing fixes from PRODUCTION_DEPLOYMENT_AUDIT.md**

---

## Priority 1: Fix Distribution Shift (CRITICAL)

### Option A: Retrain on Cropped Hands (RECOMMENDED)

#### Step 1: Create Cropped Dataset
```bash
# Run preprocessing script
python scripts/test_reproducibility.py --test 1 \
    --test-dir notebooks/data/hagrid_subset/test \
    --cropped-dir data/hagrid_subset_cropped/test

# Manually process train/val if needed, or extend script
```

#### Step 2: Modify Training Notebook
**File:** `notebooks/finetune_vit_hagrid.ipynb`

**Cell: Dataset Path (around line 101)**
```python
# OLD
DATASET_ROOT = Path('data/hagrid_subset')

# NEW
DATASET_ROOT = Path('data/hagrid_subset_cropped')  # ← Use cropped dataset
```

#### Step 3: Retrain
- Run notebook end-to-end
- Monitor accuracy on cropped test set
- Target: >85% accuracy (may be lower than 92%, but production will match)

---

## Priority 2: Fix Aspect Ratio Handling

### Option A: Add Square Padding to Training (RECOMMENDED)

**File:** `notebooks/finetune_vit_hagrid.ipynb`

**Cell: Transform Definition (around line 100)**
```python
from torchvision import transforms
from PIL import Image

class SquarePad:
    """Pad image to square aspect ratio before resize."""
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        
        # Pad to square
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return transforms.functional.pad(img, padding, fill=0)

# Update transform
transform = transforms.Compose([
    SquarePad(),  # ← ADD THIS
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Priority 3: Fix Interpolation Method

### Update HandCropper

**File:** `src/recognition_engine/hand_detection/hand_cropper.py`

**Line 306 (in _preprocess method)**
```python
# OLD
processed = cv2.resize(image, self._target_size, 
                      interpolation=cv2.INTER_LINEAR)

# NEW
processed = cv2.resize(image, self._target_size, 
                      interpolation=cv2.INTER_CUBIC)  # ← Match training (PIL bicubic)
```

---

## Priority 4: Add Temporal Smoothing

### Implementation 1: Simple Voting

**File:** `scripts/demo_gesture_recognition.py`

**Add new class:**
```python
from collections import deque, Counter

class TemporalSmoother:
    """Smooth predictions over time using voting."""
    
    def __init__(self, window_size=5, min_confidence=0.6, stability_frames=3):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        
        self.history = deque(maxlen=window_size)
        self.current_gesture = None
        self.stable_count = 0
    
    def update(self, gesture: str, confidence: float) -> tuple:
        """
        Update history and return smoothed prediction.
        
        Returns:
            (smoothed_gesture, smoothed_confidence)
        """
        # Add to history
        self.history.append((gesture, confidence))
        
        # Filter by confidence
        valid_predictions = [
            (g, c) for g, c in self.history if c >= self.min_confidence
        ]
        
        if not valid_predictions:
            return self.current_gesture or "unknown", 0.0
        
        # Vote for most common gesture
        gesture_votes = Counter(g for g, c in valid_predictions)
        top_gesture, vote_count = gesture_votes.most_common(1)[0]
        
        # Calculate average confidence for top gesture
        confidences = [c for g, c in valid_predictions if g == top_gesture]
        avg_confidence = np.mean(confidences)
        
        # Require stability before changing gesture
        if top_gesture == self.current_gesture:
            self.stable_count += 1
        else:
            if self.stable_count < self.stability_frames:
                # Not stable enough, keep current gesture
                return self.current_gesture or top_gesture, avg_confidence
            
            # Switch to new gesture
            self.current_gesture = top_gesture
            self.stable_count = 0
        
        return self.current_gesture, avg_confidence
```

**Modify GestureRecognitionDemo.__init__:**
```python
def __init__(self, ...):
    # ... existing code ...
    self.smoother = TemporalSmoother(
        window_size=5, 
        min_confidence=0.6,
        stability_frames=3
    )
```

**Modify process_frame method:**
```python
def process_frame(self, frame: np.ndarray) -> np.ndarray:
    detections = self.detector.detect(frame)
    
    for detection in detections:
        # ... existing cropping code ...
        
        # Raw prediction
        gesture, confidence = self.classify_gesture(hand_roi.image)
        
        # Smooth over time
        smooth_gesture, smooth_confidence = self.smoother.update(
            gesture, confidence
        )
        
        # Display smoothed prediction
        if smooth_confidence >= self.confidence_threshold:
            # Draw bbox and label with smooth_gesture, smooth_confidence
            label = f"{smooth_gesture}: {smooth_confidence:.2f}"
            # ... rest of visualization code ...
    
    return frame
```

---

## Priority 5: Add Confidence Diagnostics

### Add Logging to Demo

**File:** `scripts/demo_gesture_recognition.py`

**Add to classify_gesture method:**
```python
def classify_gesture(self, hand_roi: np.ndarray) -> tuple:
    pil_img = Image.fromarray(hand_roi)
    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
    
    with torch.no_grad():
        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    pred_idx = output.argmax(dim=1).item()
    confidence = probs[pred_idx].item()
    gesture = GESTURE_CLASSES[pred_idx]
    
    # ADD DIAGNOSTIC LOGGING
    if hasattr(self, 'debug_mode') and self.debug_mode:
        top5_probs, top5_idx = probs.topk(5)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        
        logger.debug(f"Prediction: {gesture} ({confidence:.3f})")
        logger.debug(f"Top-5: {[(GESTURE_CLASSES[i], p.item()) for i, p in zip(top5_idx, top5_probs)]}")
        logger.debug(f"Entropy: {entropy:.3f}")
    
    return gesture, confidence
```

**Add command-line flag:**
```python
# In main()
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args = parser.parse_args()

demo = GestureRecognitionDemo(...)
demo.debug_mode = args.debug
```

**Usage:**
```bash
python scripts/demo_gesture_recognition.py --debug
```

---

## Quick Test Workflow

### 1. Validate Current Performance
```bash
# Measure accuracy drop on cropped hands
python scripts/test_reproducibility.py --test 1
```

### 2. Apply Quick Fixes
- Fix interpolation (HandCropper)
- Add temporal smoothing (demo script)
- Add debug logging

### 3. Test Quick Fixes
```bash
# Run demo with debug mode
python scripts/demo_gesture_recognition.py --debug --confidence 0.6
```

### 4. Prepare for Retraining
```bash
# Create cropped dataset for train/val/test
# (Extend test_reproducibility.py or write custom script)
python scripts/preprocess_dataset.py \
    --source-dir notebooks/data/hagrid_subset \
    --output-dir data/hagrid_subset_cropped
```

### 5. Retrain Model
- Open `notebooks/finetune_vit_hagrid.ipynb`
- Update `DATASET_ROOT` to cropped dataset
- Run training cells
- Monitor accuracy (target >85%)

### 6. Deploy Retrained Model
```bash
# Copy new checkpoint
cp notebooks/checkpoints/best_model.pth checkpoints/best_model_cropped.pth

# Test production
python scripts/demo_gesture_recognition.py \
    --checkpoint checkpoints/best_model_cropped.pth
```

---

## Verification Checklist

### After Quick Fixes (Priority 2-5)
- [ ] Interpolation method changed to INTER_CUBIC
- [ ] Temporal smoothing implemented
- [ ] Demo runs without flickering
- [ ] Confidence logs show reasonable distributions
- [ ] Subjective improvement in demo stability

### After Retraining (Priority 1)
- [ ] Cropped dataset created for train/val/test
- [ ] Training converges (>85% accuracy on cropped test set)
- [ ] Confusion matrix shows good diagonality
- [ ] Demo with new model shows stable predictions
- [ ] Production accuracy within 5% of test accuracy

### Final Validation
- [ ] Run full reproducibility test suite
- [ ] Compare original vs retrained model
- [ ] Document accuracy improvements
- [ ] Update deployment documentation

---

## Rollback Plan

If retraining doesn't improve production performance:

1. **Revert to original model**
   ```bash
   cp checkpoints/best_model_original.pth checkpoints/best_model.pth
   ```

2. **Keep quick fixes** (temporal smoothing, interpolation)

3. **Investigate alternative hypotheses:**
   - Check for hand detection failures
   - Validate lighting/background conditions
   - Test with different webcam hardware
   - Analyze failure modes (specific gestures)

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Validation | 1 day | Reproducibility test results |
| Quick fixes | 1-2 days | Updated demo with smoothing |
| Dataset prep | 2-3 days | Cropped HaGRID dataset |
| Retraining | 2-4 hours | New model checkpoint |
| Validation | 1 day | Production performance metrics |
| **Total** | **1 week** | Production-ready model |

---

## Success Metrics

### Before Fixes
- Test accuracy (full-frame): 92.49%
- Test accuracy (cropped): ~65% (estimated)
- Production accuracy: Poor (subjective)
- Confidence: Unknown distribution
- Stability: Flickering, unstable

### After Quick Fixes
- Production stability: Improved (temporal smoothing)
- Confidence: Logged and analyzed
- User experience: Noticeably better

### After Retraining
- Test accuracy (cropped): >85%
- Production accuracy: 80-90%
- Gap (test vs prod): <5%
- Confidence (correct predictions): >0.75 mean
- Stability: Smooth, reliable
- Per-class recall: >70% for all gestures

---

## Troubleshooting

### If accuracy doesn't improve after retraining:
1. Verify cropped dataset was created correctly
2. Check that dataset path is updated in notebook
3. Validate transforms match between training and production
4. Inspect sample images from cropped dataset
5. Check for data leakage or class imbalance

### If temporal smoothing causes lag:
1. Reduce window_size (5 → 3)
2. Reduce stability_frames (3 → 2)
3. Lower min_confidence (0.6 → 0.5)

### If confidence is still low after all fixes:
1. Check for systematic preprocessing bugs
2. Validate model checkpoint loaded correctly
3. Test with known-good images from test set
4. Consider model architecture changes (larger model, different backbone)

---

**Document Status:** Ready for implementation  
**Owner:** ML Engineer  
**Last Updated:** January 17, 2026

See [PRODUCTION_DEPLOYMENT_AUDIT.md](PRODUCTION_DEPLOYMENT_AUDIT.md) for complete analysis.
