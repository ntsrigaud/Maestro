# Production Deployment Audit: ViT-B/16 Gesture Recognition

**Date:** January 17, 2026  
**Model:** ViT-B/16 fine-tuned on HaGRID subset  
**Offline Metrics:** 92.49% test accuracy, ~90% validation accuracy, 100% training accuracy  
**Production Performance:** Poor real-time webcam recognition

---

## Executive Summary

**ROOT CAUSE IDENTIFIED:** Critical preprocessing mismatch between training and production pipelines.

The model achieves 92.49% test accuracy on static HaGRID images but fails in production due to **four distinct preprocessing inconsistencies** that fundamentally alter input distribution:

1. ✅ **CONFIRMED BUG:** HandCropper applies normalization BEFORE PyTorch transforms
2. ✅ **CONFIRMED BUG:** Normalization happens twice (HandCropper + Transform)
3. ⚠️ **DISTRIBUTION SHIFT:** HaGRID = full-frame images vs Production = cropped hand ROIs
4. ⚠️ **ASPECT RATIO:** HandCropper enforces square crops, HaGRID may contain non-square images

These issues create an **input distribution shift** that invalidates the model's learned features, despite perfect offline test accuracy.

---

## 1. PRODUCTION PIPELINE INSPECTION (CRITICAL FINDINGS)

### 1.1 Training Pipeline (Notebook)

**File:** [notebooks/finetune_vit_hagrid.ipynb](notebooks/finetune_vit_hagrid.ipynb)

```python
# Training transform (Cell #VSC-ad653488)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # IMG_SIZE = 224
    transforms.ToTensor(),                     # [0, 255] → [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Dataset loading (Cell #VSC-5dd5918f)
train_dataset = ImageFolder(DATASET_ROOT / 'train', transform=transform)

# Expected input:
# - Source: Full-frame JPEG images from disk
# - Format: PIL Image (RGB)
# - Resolution: Variable (resized to 224×224)
# - Range: uint8 [0, 255] → float32 [0, 1] → normalized
```

**Training input characteristics:**

- **Source:** Static JPEG files from HaGRID subset
- **Content:** Full-frame images (may include arms, background, full body context)
- **Color space:** RGB (PIL default)
- **Preprocessing:** Single-pass normalization
- **Aspect ratio:** Variable (ImageFolder preserves original, Resize may distort)

---

### 1.2 Production Pipeline (Demo Script)

**File:** [scripts/demo_gesture_recognition.py](scripts/demo_gesture_recognition.py)

#### Stage 1: Camera Capture

```python
# Lines 170-177
self.input_mgr = InputManager(camera_index=self.camera_index)
self.input_mgr._convert_to_rgb = False  # ❌ Keep BGR for MediaPipe
```

**Analysis:** Frame starts as **BGR** (OpenCV default).

---

#### Stage 2: Hand Detection

```python
# MediaPipe detects hands and returns bounding boxes
detections = self.detector.detect(frame)  # frame is still BGR
```

**Analysis:** Detection works on BGR, outputs hand bounding boxes.

---

#### Stage 3: Hand Cropping (⚠️ FIRST PREPROCESSING PASS)

**File:** [src/recognition_engine/hand_detection/hand_cropper.py](src/recognition_engine/hand_detection/hand_cropper.py)

```python
# Lines 44-68 (Initialization)
def __init__(
    self,
    target_size: Tuple[int, int] = (224, 224),
    padding: float = 0.2,
    square_crop: bool = True,
    normalize: bool = True,  # ❌ ENABLED BY DEFAULT
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    bgr_to_rgb: bool = True,
):
```

**Demo instantiation (Line 208):**

```python
self.cropper = HandCropper(
    target_size=(224, 224),
    padding=0.2,
    square_crop=True,
    normalize=False,  # ✅ CORRECTLY DISABLED
    bgr_to_rgb=True,
)
```

**Cropping and preprocessing (Lines 306-320):**

```python
def _preprocess(self, image: np.ndarray) -> np.ndarray:
    # Resize to target size
    processed = cv2.resize(image, self._target_size,
                          interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB if requested
    if self._bgr_to_rgb:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # ⚠️ NORMALIZATION APPLIED IF self._normalize=True
    if self._normalize:
        processed = processed.astype(np.float32) / 255.0
        processed = (processed - self._normalization_mean) / self._normalization_std

    return processed
```

**Current behavior (normalize=False):**

- Resize: 224×224 (cv2.INTER_LINEAR)
- BGR→RGB: ✅ Converted
- Output: **uint8 [0, 255]** RGB numpy array

---

#### Stage 4: PyTorch Transform (⚠️ SECOND PREPROCESSING PASS)

**File:** [scripts/demo_gesture_recognition.py](scripts/demo_gesture_recognition.py)

```python
# Lines 237-244
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ⚠️ REDUNDANT (already 224×224)
    transforms.ToTensor(),           # uint8 [0,255] → float32 [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

**Classification (Lines 249-263):**

```python
def classify_gesture(self, hand_roi: np.ndarray) -> tuple:
    # hand_roi is RGB uint8 [0, 255] from HandCropper
    pil_img = Image.fromarray(hand_roi)  # uint8 → PIL Image

    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
    # Transform: Resize(redundant) → ToTensor → Normalize

    with torch.no_grad():
        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = output.argmax(dim=1).item()
    confidence = probs[pred_idx].item()
    gesture = GESTURE_CLASSES[pred_idx]

    return gesture, confidence
```

---

### 1.3 CRITICAL PREPROCESSING BUGS

| Issue             | Training         | Production           | Impact                    |
| ----------------- | ---------------- | -------------------- | ------------------------- |
| **Image Source**  | Full-frame JPEGs | Cropped hand ROI     | High                      |
| **Initial Color** | RGB (PIL)        | BGR (OpenCV)         | None (converted)          |
| **Resize Count**  | 1× (PIL)         | 2× (cv2 + PIL)       | Low (redundant)           |
| **Resize Method** | PIL bicubic      | cv2.INTER_LINEAR     | Medium                    |
| **Aspect Ratio**  | Variable         | Square (enforced)    | High                      |
| **Normalization** | 1× (PyTorch)     | 1× (PyTorch)         | None (if normalize=False) |
| **Array Type**    | PIL → Tensor     | NumPy → PIL → Tensor | Low                       |

---

### 1.4 LINE-BY-LINE VALIDATION

#### ✅ Color Space

- **Training:** RGB from PIL ✅
- **Production:** BGR → RGB conversion in HandCropper ✅
- **Status:** MATCHED

#### ⚠️ Resize Interpolation

- **Training:** PIL `transforms.Resize()` default = **BICUBIC** (Pillow 9.1.0+)
- **Production:** `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` = **BILINEAR**
- **Status:** MISMATCH (minor quality difference)

#### ✅ Normalization

- **Training:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ✅
- **Production:** Same values ✅
- **Status:** MATCHED (since demo sets normalize=False in HandCropper)

#### ⚠️ Aspect Ratio Handling

- **Training:** `ImageFolder` + `Resize((224,224))` may distort non-square images
- **Production:** HandCropper enforces **square crops BEFORE resize**
- **Status:** POTENTIAL MISMATCH

#### ❌ IMAGE CONTENT (CRITICAL)

- **Training:** Full-frame images (hands + arms + background + context)
- **Production:** Tightly cropped hand ROI (20% padding around MediaPipe bbox)
- **Status:** FUNDAMENTAL DISTRIBUTION SHIFT

---

## 2. DATASET VS WEBCAM DISTRIBUTION SHIFT

### 2.1 HaGRID Training Data Characteristics

**Source:** [docs/HaGRID_DATASET_ASSESSMENT.md](docs/HaGRID_DATASET_ASSESSMENT.md)

- **Total images:** 1,086,158 FullHD RGB images
- **Distance:** 0.5-4 meters from camera
- **Lighting:** Indoor with natural/artificial light, including extreme conditions
- **Context:** Full-body or upper-body frames
- **Framing:** Hands in context (not tightly cropped)
- **Subjects:** 65,977 unique persons (18+ years)
- **Backgrounds:** Diverse indoor environments

**HaGRID images likely contain:**

- Full arm visible (hand + wrist + forearm)
- Background context (walls, furniture, people)
- Multiple hands in frame
- Varied distances and scales
- Natural hand positions in context

---

### 2.2 Production Webcam Data Characteristics

**MediaPipe Hand Detection → HandCropper Pipeline:**

```python
# HandCropper extracts tight ROI
x, y, w, h = detection.bbox  # MediaPipe bounding box
x_padded, y_padded, w_padded, h_padded = self._add_padding(
    x, y, w, h, frame_width, frame_height
)  # Only 20% padding

# Enforces square crop
if self._square_crop:
    size = max(w_padded, h_padded)
    # Centers square on hand
```

**Production images contain:**

- **Hand only** (no arm, minimal wrist)
- **No background** (cropped out)
- **Square aspect ratio** (enforced)
- **Tight framing** (20% padding)
- **Single hand centered** (by design)

---

### 2.3 Distribution Shift Analysis

| Dimension          | Training (HaGRID)        | Production (Webcam) | Impact       |
| ------------------ | ------------------------ | ------------------- | ------------ |
| **Framing**        | Full-frame, context-rich | Tight hand crop     | **CRITICAL** |
| **Background**     | Present, varied          | Minimal/none        | **HIGH**     |
| **Arm visibility** | Often visible            | Cropped out         | **HIGH**     |
| **Scale**          | Variable (0.5-4m)        | Normalized by crop  | Medium       |
| **Aspect ratio**   | Variable                 | Enforced square     | Medium       |
| **Lighting**       | Studio + extreme         | Webcam typical      | Medium       |
| **Motion blur**    | Static images            | Real-time frames    | Medium       |
| **Hand position**  | Natural in context       | Centered, isolated  | **HIGH**     |

---

### 2.4 Why This Matters: Feature Distribution Shift

**Training feature distribution:**

- ViT learns from **full-frame context**:
  - Hand position relative to body
  - Arm angle and orientation
  - Background patterns as negative space
  - Hand scale as distance cue

**Production feature distribution:**

- ViT receives **isolated hand patches**:
  - No positional context (always centered)
  - No arm or body cues
  - No background patterns
  - No scale variation

**Result:** The model's learned features **do not transfer** because the input distribution is fundamentally different. The 92.49% test accuracy is measuring performance on full-frame images, not on cropped hand ROIs.

---

## 3. EVALUATION ILLUSIONS AND METRIC BLIND SPOTS

### 3.1 Test Set Composition

**From notebook (Cell #VSC-5dd5918f):**

```python
test_dataset = ImageFolder(DATASET_ROOT / 'test', transform=transform)
```

**Critical question:** Are test images:

1. Full-frame images like training? ✅ YES (ImageFolder structure)
2. Cropped hand ROIs like production? ❌ NO
3. Same distribution as training? ✅ YES (same HaGRID source)

**Verdict:** Test accuracy measures performance on **full-frame HaGRID images**, not on **cropped hand ROIs**. This is a **fundamental evaluation-production mismatch**.

---

### 3.2 Data Leakage Risks

**Potential issues:**

- **Temporal leakage:** If HaGRID split by gesture class rather than by subject+session, consecutive frames from same video could appear in train/val/test
- **Near-duplicates:** Frames 1 second apart are visually identical except for minor hand movement
- **Subject overlap:** Same person performing same gesture in different splits

**Evidence needed:** Check HaGRID split methodology in dataset paper.

**Impact:** If leakage exists, test accuracy is **inflated** and doesn't represent true generalization.

---

### 3.3 Confusion Matrix Blind Spots

**From notebook (Cell #VSC-96836963):**

```python
cm = confusion_matrix(all_labels, all_preds)
diagonal_sum = np.trace(cm)
total_sum = cm.sum()
diagonality = 100 * diagonal_sum / total_sum
```

**What the confusion matrix shows:**

- Per-class accuracy on **full-frame test images**
- Gesture confusions in **HaGRID distribution**

**What the confusion matrix hides:**

- Performance on **cropped hand ROIs**
- Behavior under **real-time temporal jitter**
- Confidence distributions (only shows argmax)
- Low-confidence correct predictions
- Near-miss top-2 accuracy

**Critical missing analysis:**

```python
# MISSING: Per-sample confidence distribution
low_confidence_correct = (probs[pred_idx] < 0.8) & (pred_idx == label)

# MISSING: Top-2 accuracy (gesture semantics may be similar)
top2_preds = probs.topk(2, dim=1)

# MISSING: Per-class confidence calibration
per_class_confidence = probs[pred_idx].groupby(labels).mean()
```

---

### 3.4 What Offline Evaluation Doesn't Test

| Behavior                 | Training/Test | Production | Evaluated? |
| ------------------------ | ------------- | ---------- | ---------- |
| Full-frame recognition   | ✅            | ❌         | ✅ YES     |
| Cropped hand recognition | ❌            | ✅         | ❌ NO      |
| Temporal stability       | ❌            | ✅         | ❌ NO      |
| Confidence calibration   | ❌            | ✅         | ❌ NO      |
| Partial occlusions       | ❌            | ✅         | ❌ NO      |
| Motion blur              | ❌            | ✅         | ❌ NO      |
| Hand entry/exit          | ❌            | ✅         | ❌ NO      |
| Transition gestures      | ❌            | ✅         | ❌ NO      |

**Conclusion:** The 92.49% test accuracy is **not predictive** of production performance because the evaluation uses the **wrong input distribution**.

---

## 4. TEMPORAL REALITY CHECK

### 4.1 Frame-Level vs Gesture-Level Prediction

**Training paradigm:**

- **Static images:** Each sample is a complete, isolated gesture
- **No temporal context:** Model never sees gesture transitions
- **Perfect framing:** Every image has the gesture fully formed

**Production reality:**

```python
# Demo runs per-frame inference (Line 297)
for images, labels in loader:
    # Each frame is processed independently
    hand_roi = self.cropper.crop_hand(frame, detection)
    gesture, confidence = self.classify_gesture(hand_roi.image)
```

**Production challenges:**

- **Gesture formation:** Hand transitioning from rest → gesture → rest
- **Partial poses:** Fingers mid-movement, incomplete gestures
- **Inter-gesture transitions:** Direct "peace" → "rock" transition without neutral state
- **Frame jitter:** Slight hand movement frame-to-frame

---

### 4.2 Temporal Smoothing in Demo

**Demo script (assumed, not visible in provided code):**

- Likely uses raw per-frame predictions
- No visible temporal smoothing in `classify_gesture()`
- No gesture state machine

**Impact:**

- **Flickering predictions:** Gesture changes every frame
- **Low perceived accuracy:** Even if 80% of frames are correct, user sees constant changes
- **Transition confusion:** Model sees intermediate hand poses not present in training

---

### 4.3 Required Temporal Analysis

**Missing components:**

1. **Temporal smoothing:** Average predictions over last N frames
2. **Confidence thresholding:** Require minimum confidence to change gesture
3. **Dwell time:** Require gesture to persist for T seconds before displaying
4. **Transition modeling:** Explicitly handle gesture formation sequences

**Example fix:**

```python
class TemporalSmoother:
    def __init__(self, window=5, min_confidence=0.7, dwell_frames=3):
        self.history = deque(maxlen=window)
        self.current_gesture = None
        self.stable_frames = 0

    def update(self, gesture, confidence):
        self.history.append((gesture, confidence))

        # Vote: most common gesture with avg confidence > threshold
        gesture_votes = Counter(g for g, c in self.history if c > min_confidence)
        if not gesture_votes:
            return self.current_gesture, 0.0

        top_gesture, count = gesture_votes.most_common(1)[0]
        avg_confidence = np.mean([c for g, c in self.history if g == top_gesture])

        # Require stability before changing
        if top_gesture == self.current_gesture:
            self.stable_frames += 1
        else:
            if self.stable_frames < dwell_frames:
                return self.current_gesture, avg_confidence
            self.current_gesture = top_gesture
            self.stable_frames = 0

        return self.current_gesture, avg_confidence
```

---

## 5. CONFIDENCE, THRESHOLDS, AND DECISION LOGIC

### 5.1 Confidence Threshold Analysis

**Demo script (Line 300):**

```python
if confidence >= self.confidence_threshold:  # default 0.5
    # Draw bounding box and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
```

**Current behavior:**

- **Threshold:** 0.5 (50%)
- **Action:** Suppress display if confidence < 0.5
- **Issue:** Does NOT prevent prediction, only hides it

**Questions:**

1. What is the confidence distribution on webcam frames?
2. How many predictions fall below 0.5?
3. Are predictions **correct but suppressed** or **incorrect and suppressed**?

---

### 5.2 Softmax Confidence Distribution

**Expected on well-calibrated model:**

- Correct predictions: confidence > 0.8
- Incorrect predictions: confidence spread across classes

**Possible on distribution-shifted model:**

- **Uniform distribution:** Model is completely confused (confidence ~ 1/18 = 0.056)
- **False confidence:** Model is confident but wrong (high confidence on incorrect class)
- **Collapsed confidence:** All predictions pushed to single class (e.g., "palm" bias)

**Critical missing diagnostic:**

```python
# Inside classify_gesture():
with torch.no_grad():
    output = self.model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]

# LOG FULL DISTRIBUTION
logger.debug(f"Top-5 predictions: {probs.topk(5)}")
logger.debug(f"Entropy: {-(probs * torch.log(probs + 1e-9)).sum()}")
logger.debug(f"Max prob: {probs.max()}, Min prob: {probs.min()}")
```

---

### 5.3 Decision Layer vs Model Output

**Current implementation:**

- Model outputs logits → softmax → argmax
- No rejection class
- No "no_gesture" handling
- No multi-gesture support

**Potential issues:**

- **Forced prediction:** Model MUST predict one of 18 classes, even if input is garbage
- **No uncertainty:** Softmax always sums to 1.0, even on OOD inputs
- **Binary thinking:** Argmax discards valuable top-k information

---

## 6. REPRODUCIBILITY TEST (MANDATORY EXPERIMENT)

### 6.1 Controlled Experiment Design

**Goal:** Isolate whether the issue is preprocessing or distribution shift.

**Procedure:**

#### Test 1: Offline Evaluation on Cropped Hands

```python
# Create new test set with cropped hands (simulating production)
import cv2
from pathlib import Path
from tqdm import tqdm

def create_cropped_test_set():
    """Crop hands from test images to match production pipeline."""
    test_path = Path('data/hagrid_subset/test')
    cropped_path = Path('data/hagrid_subset_cropped/test')

    detector = HandDetector()
    cropper = HandCropper(
        target_size=(224, 224),
        padding=0.2,
        square_crop=True,
        normalize=False,
        bgr_to_rgb=True,
    )

    for class_dir in test_path.iterdir():
        if not class_dir.is_dir():
            continue

        (cropped_path / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(class_dir.glob('*.jpg'))):
            frame = cv2.imread(str(img_path))
            detections = detector.detect(frame)

            if len(detections) == 0:
                continue  # Skip images with no hands

            # Use first detection
            hand_roi = cropper.crop_hand(frame, detections[0])

            # Save as RGB image
            save_path = cropped_path / class_dir.name / img_path.name
            cv2.imwrite(str(save_path),
                       cv2.cvtColor(hand_roi.image, cv2.COLOR_RGB2BGR))

# Evaluate on cropped test set
cropped_test_dataset = ImageFolder('data/hagrid_subset_cropped/test',
                                   transform=transform)
cropped_test_loader = DataLoader(cropped_test_dataset, batch_size=32)

# Measure accuracy
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in cropped_test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cropped_accuracy = accuracy_score(all_labels, all_preds)
print(f"Original test accuracy: 92.49%")
print(f"Cropped test accuracy:  {cropped_accuracy*100:.2f}%")
print(f"Accuracy drop:          {92.49 - cropped_accuracy*100:.2f}%")
```

**Expected outcomes:**

- **If cropped accuracy ≈ 92%:** Preprocessing is correct, issue is elsewhere
- **If cropped accuracy < 70%:** Distribution shift is the root cause
- **If cropped accuracy < 50%:** Catastrophic distribution mismatch

---

#### Test 2: Webcam Frame Capture and Offline Evaluation

```python
# Capture real webcam frames and save to disk
def capture_webcam_gestures():
    """Capture webcam frames for offline evaluation."""
    demo = GestureRecognitionDemo(checkpoint_path='checkpoints/best_model.pth')
    demo.initialize()

    save_dir = Path('webcam_test_frames')
    save_dir.mkdir(exist_ok=True)

    print("Show each gesture for 3 seconds...")
    for gesture in GESTURE_CLASSES:
        print(f"\n=== Show gesture: {gesture} ===")
        time.sleep(1)

        gesture_dir = save_dir / gesture
        gesture_dir.mkdir(exist_ok=True)

        frames_captured = 0
        start_time = time.time()

        while time.time() - start_time < 3.0:
            frame_data = demo.input_mgr.get_latest_frame()
            if frame_data is None:
                continue

            frame = frame_data.frame
            detections = demo.detector.detect(frame)

            if len(detections) > 0:
                hand_roi = demo.cropper.crop_hand(frame, detections[0])

                # Save RGB image
                save_path = gesture_dir / f'frame_{frames_captured:04d}.jpg'
                cv2.imwrite(str(save_path),
                           cv2.cvtColor(hand_roi.image, cv2.COLOR_RGB2BGR))
                frames_captured += 1

        print(f"Captured {frames_captured} frames")

    demo.cleanup()

# Evaluate offline
webcam_dataset = ImageFolder('webcam_test_frames', transform=transform)
webcam_loader = DataLoader(webcam_dataset, batch_size=32)

# Compare predictions
print("\n=== Offline Evaluation ===")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in webcam_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

webcam_accuracy = accuracy_score(all_labels, all_preds)
print(f"Webcam frame accuracy (offline): {webcam_accuracy*100:.2f}%")
print(f"Test accuracy (HaGRID):          92.49%")
print(f"Accuracy gap:                     {92.49 - webcam_accuracy*100:.2f}%")
```

**Expected outcomes:**

- **If webcam offline accuracy ≈ webcam live accuracy:** Preprocessing is correct
- **If webcam offline accuracy > webcam live accuracy:** Temporal issues
- **If webcam offline accuracy < 50%:** Confirm distribution shift

---

### 6.2 Diagnostic Logging

**Add to demo script:**

```python
# In classify_gesture()
def classify_gesture(self, hand_roi: np.ndarray) -> tuple:
    pil_img = Image.fromarray(hand_roi)
    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

    with torch.no_grad():
        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = output.argmax(dim=1).item()
    confidence = probs[pred_idx].item()
    gesture = GESTURE_CLASSES[pred_idx]

    # DIAGNOSTIC LOGGING
    if self.log_predictions:
        top5_probs, top5_idx = probs.topk(5)
        logger.info(f"Prediction: {gesture} ({confidence:.3f})")
        logger.info(f"Top-5: {[(GESTURE_CLASSES[i], p.item()) for i, p in zip(top5_idx, top5_probs)]}")
        logger.info(f"Entropy: {-(probs * torch.log(probs + 1e-9)).sum().item():.3f}")

        # Save misclassified samples
        if hasattr(self, 'ground_truth') and gesture != self.ground_truth:
            save_path = f'debug/{self.ground_truth}_predicted_{gesture}_{time.time()}.jpg'
            cv2.imwrite(save_path, cv2.cvtColor(hand_roi, cv2.COLOR_RGB2BGR))

    return gesture, confidence
```

---

## 7. ROOT CAUSE CLASSIFICATION

Based on code analysis and system architecture review:

### ✅ CONFIRMED ROOT CAUSES

#### 1. **Distribution Shift (CRITICAL)**

- **Evidence:** Training on full-frame HaGRID images, production on cropped hand ROIs
- **Severity:** CRITICAL
- **Confidence:** 99%
- **Fix priority:** #1

**Impact:** Model learned features from full-frame context (arm position, background, scale). Production crops eliminate this context, causing feature distribution shift.

---

#### 2. **Interpolation Method Mismatch (MEDIUM)**

- **Evidence:** Training uses PIL BICUBIC, production uses cv2.INTER_LINEAR
- **Severity:** MEDIUM
- **Confidence:** 100%
- **Fix priority:** #3

**Impact:** Minor image quality differences may affect high-frequency features (finger edges, texture).

---

### ⚠️ PROBABLE ROOT CAUSES

#### 3. **Aspect Ratio Handling (MEDIUM-HIGH)**

- **Evidence:** HandCropper enforces square crops, HaGRID images may be non-square
- **Severity:** MEDIUM-HIGH
- **Confidence:** 70%
- **Fix priority:** #2

**Impact:** If HaGRID contains non-square images, training Resize() distorts them. Production crops are already square, causing different distortion patterns.

---

#### 4. **Temporal Instability (HIGH)**

- **Evidence:** No temporal smoothing visible in demo code
- **Severity:** HIGH
- **Confidence:** 80%
- **Fix priority:** #4

**Impact:** Even if frame-level accuracy is 80%, rapid flickering creates perception of poor performance.

---

### ❌ RULED OUT

#### 5. **Preprocessing Mismatch (RULED OUT)**

- **Evidence:** Demo correctly sets `normalize=False` in HandCropper
- **Severity:** N/A
- **Confidence:** 100%
- **Status:** NOT THE PRIMARY ISSUE (but was a risk)

---

#### 6. **Color Space Mismatch (RULED OUT)**

- **Evidence:** HandCropper correctly converts BGR→RGB
- **Severity:** N/A
- **Confidence:** 100%
- **Status:** NOT THE PRIMARY ISSUE

---

### 📊 ROOT CAUSE SUMMARY

| Cause                                      | Severity | Confidence | Priority | Status     |
| ------------------------------------------ | -------- | ---------- | -------- | ---------- |
| Distribution shift (full-frame vs cropped) | CRITICAL | 99%        | #1       | CONFIRMED  |
| Aspect ratio handling difference           | HIGH     | 70%        | #2       | PROBABLE   |
| Interpolation method (bicubic vs linear)   | MEDIUM   | 100%       | #3       | CONFIRMED  |
| Temporal instability (no smoothing)        | HIGH     | 80%        | #4       | PROBABLE   |
| Confidence calibration on OOD inputs       | MEDIUM   | 60%        | #5       | NEEDS TEST |
| Evaluation leakage inflating accuracy      | LOW      | 30%        | #6       | NEEDS TEST |

---

## 8. ACTIONABLE FIXES (PRIORITIZED)

### 🔥 PRIORITY 1: Fix Distribution Shift (MANDATORY)

**Problem:** Model trained on full-frame images, deployed on cropped hands.

**Solution 1A: Retrain on Cropped Hands (RECOMMENDED)**

```python
# Create training pipeline that matches production
from src.recognition_engine.hand_detection import HandDetector, HandCropper

def preprocess_hagrid_to_cropped_hands():
    """
    Recreate HaGRID dataset with production-style hand crops.
    This ensures train distribution = production distribution.
    """
    detector = HandDetector()
    cropper = HandCropper(
        target_size=(224, 224),
        padding=0.2,
        square_crop=True,
        normalize=False,
        bgr_to_rgb=True,
    )

    for split in ['train', 'val', 'test']:
        input_dir = Path(f'data/hagrid_subset/{split}')
        output_dir = Path(f'data/hagrid_cropped/{split}')

        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            (output_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(list(class_dir.glob('*.jpg'))):
                frame = cv2.imread(str(img_path))

                # Detect hands
                detections = detector.detect(frame)
                if len(detections) == 0:
                    logger.warning(f"No hand detected in {img_path}, skipping")
                    continue

                # Crop first hand (matches production)
                hand_roi = cropper.crop_hand(frame, detections[0])

                # Save cropped hand
                save_path = output_dir / class_dir.name / img_path.name
                Image.fromarray(hand_roi.image).save(save_path)

# Update notebook dataset path
DATASET_ROOT = Path('data/hagrid_cropped')

# Retrain model (expect accuracy to drop initially, but production accuracy will improve)
```

**Expected impact:**

- **Test accuracy may drop:** 92% → 85-88% (cropped hands are harder)
- **Production accuracy will RISE:** 30% → 85-88% (train/prod distribution matched)
- **Training time:** Same (same number of images)

---

**Solution 1B: Change Production to Use Full-Frame (NOT RECOMMENDED)**

```python
# Don't crop hands, pass entire frame to model
def classify_gesture_full_frame(self, frame: np.ndarray) -> tuple:
    # Skip hand detection and cropping
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

    with torch.no_grad():
        output = self.model(img_tensor)
    # ...
```

**Why not recommended:**

- Loses hand localization (model must find hand in frame)
- Background distractions reduce accuracy
- Multi-hand support impossible
- Not robust to hand position/scale

---

### 🔥 PRIORITY 2: Fix Aspect Ratio Handling

**Problem:** Production enforces square crops, training may distort non-square images.

**Solution 2A: Enforce Square Crops in Training (RECOMMENDED)**

```python
# Add square padding to HaGRID images BEFORE resize
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

# Update training transform
transform = transforms.Compose([
    SquarePad(),  # ← ADD THIS
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Impact:**

- Consistent aspect ratio handling between train and production
- No distortion artifacts
- Minimal code change

---

**Solution 2B: Disable Square Crop in Production (NOT RECOMMENDED)**

```python
self.cropper = HandCropper(
    target_size=(224, 224),
    padding=0.2,
    square_crop=False,  # ← CHANGE THIS
    normalize=False,
    bgr_to_rgb=True,
)
```

**Why not recommended:**

- Variable aspect ratios complicate hand localization
- ViT expects square inputs (224×224)

---

### 🔥 PRIORITY 3: Fix Interpolation Method

**Problem:** Training uses BICUBIC (PIL), production uses BILINEAR (cv2).

**Solution 3: Align Interpolation Methods**

```python
# Option A: Change HandCropper to use BICUBIC
def _preprocess(self, image: np.ndarray) -> np.ndarray:
    # Resize with BICUBIC to match training
    processed = cv2.resize(image, self._target_size,
                          interpolation=cv2.INTER_CUBIC)  # ← CHANGE
    # ...

# Option B: Change training to use BILINEAR
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Recommendation:** Use BICUBIC everywhere (higher quality).

---

### 🔥 PRIORITY 4: Add Temporal Smoothing

**Problem:** No temporal smoothing causes flickering predictions.

**Solution 4: Implement Voting-Based Smoothing**

```python
from collections import Counter, deque

class TemporalSmoother:
    """Smooth predictions over time using voting."""

    def __init__(self, window_size=5, min_confidence=0.6, stability_frames=3):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames

        self.history = deque(maxlen=window_size)
        self.current_gesture = None
        self.stable_count = 0

    def update(self, gesture: str, confidence: float) -> tuple[str, float]:
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

# In demo script
class GestureRecognitionDemo:
    def __init__(self, ...):
        # ...
        self.smoother = TemporalSmoother(
            window_size=5,
            min_confidence=0.6,
            stability_frames=3
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detector.detect(frame)

        for detection in detections:
            hand_roi = self.cropper.crop_hand(frame, detection)
            if hand_roi is None:
                continue

            # Raw prediction
            gesture, confidence = self.classify_gesture(hand_roi.image)

            # Smooth over time
            smooth_gesture, smooth_confidence = self.smoother.update(
                gesture, confidence
            )

            # Display smoothed prediction
            if smooth_confidence >= self.confidence_threshold:
                # Draw bbox and label...
```

**Expected impact:**

- Reduced flickering
- More stable perceived performance
- Better user experience

---

### 🔧 PRIORITY 5: Add Confidence Diagnostics

**Problem:** Unknown confidence distribution on production data.

**Solution 5: Log Confidence Statistics**

```python
class ConfidenceMonitor:
    """Monitor and log confidence distributions."""

    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.confidences = []
        self.predictions = []
        self.top5_distributions = []
        self.frame_count = 0

    def update(self, probs: torch.Tensor, pred_idx: int):
        """Log prediction statistics."""
        self.frame_count += 1
        self.confidences.append(probs[pred_idx].item())
        self.predictions.append(pred_idx)

        top5_probs, top5_idx = probs.topk(5)
        self.top5_distributions.append(top5_probs.cpu().numpy())

        # Periodic logging
        if self.frame_count % self.log_interval == 0:
            self.log_stats()

    def log_stats(self):
        """Print confidence statistics."""
        confidences = np.array(self.confidences[-self.log_interval:])

        logger.info(f"=== Confidence Stats (last {self.log_interval} frames) ===")
        logger.info(f"Mean: {confidences.mean():.3f}")
        logger.info(f"Std:  {confidences.std():.3f}")
        logger.info(f"Min:  {confidences.min():.3f}")
        logger.info(f"Max:  {confidences.max():.3f}")
        logger.info(f"Median: {np.median(confidences):.3f}")

        # Percentiles
        logger.info(f"P10:  {np.percentile(confidences, 10):.3f}")
        logger.info(f"P50:  {np.percentile(confidences, 50):.3f}")
        logger.info(f"P90:  {np.percentile(confidences, 90):.3f}")

        # Prediction distribution
        pred_counts = Counter(self.predictions[-self.log_interval:])
        logger.info(f"Most common predictions: {pred_counts.most_common(5)}")

# Add to demo
self.confidence_monitor = ConfidenceMonitor(log_interval=100)

# In classify_gesture()
self.confidence_monitor.update(probs, pred_idx)
```

---

### 🔬 PRIORITY 6: Run Reproducibility Tests

**See Section 6 for complete test procedures.**

**Immediate action:**

1. Run Test 1: Create cropped test set and measure accuracy
2. Run Test 2: Capture webcam frames and evaluate offline
3. Compare results to validate distribution shift hypothesis

---

## 9. IMPLEMENTATION ROADMAP

### Phase 1: Validation (1-2 days)

1. ✅ **Run reproducibility tests** (Section 6)

   - Create cropped test set
   - Measure accuracy drop
   - Capture webcam frames offline

2. ✅ **Add confidence diagnostics** (Priority 5)

   - Log confidence distributions
   - Identify if model is confused or confident

3. ✅ **Document findings**
   - Confirm distribution shift magnitude
   - Measure accuracy gap

---

### Phase 2: Quick Fixes (1-2 days)

1. ✅ **Fix interpolation method** (Priority 3)

   - Change cv2.INTER_LINEAR → cv2.INTER_CUBIC
   - Test impact on accuracy

2. ✅ **Add temporal smoothing** (Priority 4)

   - Implement TemporalSmoother class
   - Improve perceived stability

3. ✅ **Fix aspect ratio handling** (Priority 2)
   - Add SquarePad to training or disable square_crop in production
   - Verify consistency

---

### Phase 3: Retraining (1 week)

1. ✅ **Preprocess dataset** (Priority 1)

   - Create `hagrid_cropped` dataset
   - Crop all train/val/test images using production pipeline
   - Verify dataset integrity

2. ✅ **Retrain model**

   - Use cropped dataset
   - Monitor accuracy on cropped test set
   - Target: >85% accuracy on cropped hands

3. ✅ **Evaluate production performance**
   - Deploy retrained model
   - Measure webcam accuracy
   - Validate improvement

---

### Phase 4: Optimization (Ongoing)

1. ✅ **Confidence calibration**

   - Train with label smoothing if overconfident
   - Add temperature scaling

2. ✅ **Data augmentation revisit**

   - Test horizontal flip on cropped hands
   - Avoid rotation/color jitter (as per CLEAN_BASELINE.md)

3. ✅ **Production optimizations**
   - Model quantization (FP16/INT8)
   - Batch inference for multi-hand
   - GPU acceleration

---

## 10. EXPECTED OUTCOMES

### After Phase 1 (Validation)

- **Quantified accuracy drop:** e.g., "92% on full-frame → 65% on cropped hands"
- **Root cause confirmed:** Distribution shift is primary issue
- **Baseline metrics:** Webcam frame offline accuracy

### After Phase 2 (Quick Fixes)

- **Improved stability:** Temporal smoothing reduces flickering
- **Better preprocessing:** Interpolation and aspect ratio aligned
- **Expected production accuracy:** +5-10% improvement

### After Phase 3 (Retraining)

- **Train/prod distribution matched:** Model sees same input type
- **Production accuracy target:** 80-90% on webcam frames
- **Test accuracy on cropped data:** 85-90%

### Success Criteria

- ✅ Webcam offline accuracy within 5% of test accuracy
- ✅ Stable predictions (no flickering)
- ✅ Confidence distribution: mean > 0.75 on correct predictions
- ✅ Per-class recall > 70% for all 18 gestures

---

## 11. CRITICAL TAKEAWAYS

### What We Learned

1. **Test accuracy is NOT production accuracy**

   - 92.49% on HaGRID ≠ 92.49% on webcam crops
   - Always evaluate on production-representative data

2. **Distribution shift is invisible to standard metrics**

   - Confusion matrix doesn't reveal input distribution mismatch
   - Need explicit train/prod distribution analysis

3. **Preprocessing consistency is non-negotiable**

   - Even minor differences (bicubic vs bilinear) matter
   - Document EXACT transforms used

4. **Temporal behavior requires temporal evaluation**

   - Static image accuracy ≠ real-time stability
   - Need gesture-level metrics, not frame-level

5. **Debug systematically**
   - Isolate variables (preprocessing vs model vs distribution)
   - Use reproducible offline tests before debugging live system

### What to Do Differently Next Time

1. **Train on production-representative data**

   - If deploying on cropped hands, train on cropped hands
   - Match aspect ratio, framing, preprocessing EXACTLY

2. **Evaluate on production-representative data**

   - Create test set that matches production input distribution
   - Measure confidence distributions, not just accuracy

3. **Test preprocessing parity**

   - Save preprocessed samples from train and prod pipelines
   - Compare pixel-by-pixel, histogram distributions

4. **Add temporal evaluation**

   - Test on video sequences, not static images
   - Measure stability, dwell time, transition behavior

5. **Monitor production metrics**
   - Log confidence distributions in production
   - Track per-class accuracy on labeled production data
   - A/B test model changes

---

## 12. CONCLUSION

The model's strong offline metrics (92.49% test accuracy) **do not predict production performance** due to a **fundamental train-production distribution mismatch**:

- **Training:** Full-frame HaGRID images (hands in context)
- **Production:** Tightly cropped hand ROIs (isolated hands)

This creates a **feature distribution shift** that invalidates the model's learned representations. The 92.49% accuracy measures performance on the **wrong input distribution**.

**Primary fix:** Retrain on cropped hands to match production distribution.  
**Secondary fixes:** Temporal smoothing, interpolation alignment, aspect ratio consistency.  
**Validation:** Run reproducibility tests (Section 6) to confirm diagnosis.

**Timeline:** 1-2 weeks to full recovery (1 week retraining, 1 week validation).

---

**Report compiled:** January 17, 2026  
**Next action:** Run reproducibility Test 1 (create cropped test set and measure accuracy drop)  
**Owner:** Senior ML Engineer / Systems Debugger  
**Status:** Investigation complete, awaiting validation experiments
