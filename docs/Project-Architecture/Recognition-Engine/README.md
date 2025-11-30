# Recognition Engine

The Recognition Engine is the core of the project, responsible for performing hand gesture recognition. It takes video frames from the Input Manager and outputs recognized gestures, including their semantic meaning.

## Status

### Completed Components

- **Hand Detection Submodule** - Implemented using MediaPipe Hands
  - HandDetector: Real-time hand localization with 21 landmarks
  - HandCropper: ROI extraction with preprocessing
  - Full documentation and examples available

### In Progress

- **Gesture Classification** - ViT-based static and dynamic gesture recognition
- **Semantic Understanding** - ALBEF-inspired text-gesture alignment

## Submodules

### 1. Hand Detection (Implemented)

The hand detection submodule provides real-time hand localization and region extraction.

**Components**:

- `HandDetector`: Detects hands using MediaPipe Hands, extracts 21 landmarks, computes bounding boxes
- `HandCropper`: Extracts and preprocesses hand regions for classification
- `HandDetectionResult`: Data structure for detection results (bbox, landmarks, handedness, confidence)
- `HandROI`: Data structure for cropped hand regions

**Features**:

- Multi-hand detection (up to 2 hands)
- Real-time performance (30+ FPS)
- Configurable preprocessing (resize, normalize, color conversion)
- Robust frame boundary handling

**Quick Start**:

```python
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper

input_mgr = InputManager()
detector = HandDetector(max_num_hands=2)
cropper = HandCropper(target_size=(224, 224))

input_mgr.start_continuous_capture()
frame_data = input_mgr.get_latest_frame()

# Detect hands
detections = detector.detect(frame_data.frame)

# Crop hand regions
hand_rois = cropper.crop_multiple(frame_data.frame, detections)
```

**Documentation**: [Hand Detection Documentation](HAND_DETECTION_DOCUMENTATION.md)

**Demo**: `python scripts/demo_hand_detection.py`

### 2. Gesture Classification (Planned)

The gesture classification submodule will recognize static and dynamic gestures using Vision Transformer (ViT) and LSTM.

**Planned Components**:

- Static gesture classifier (ViT-based)
- Dynamic gesture classifier (ViT + LSTM)
- Gesture vocabulary and training pipeline

### 3. Semantic Understanding (Planned)

ALBEF-inspired text-gesture alignment for mapping gestures to semantic actions.

**Planned Components**:

- Contrastive learning framework
- Text encoder (BERT-based)
- Gesture-command alignment model

## Technical Design

The Recognition Engine's design is based on the **HAVIT (Hybrid Attention-based Vision-language Interaction Transformer)** model, a state-of-the-art architecture for gesture-based interaction. This approach integrates a **Vision Transformer (ViT)** for robust visual feature extraction and **ALBEF (Align before Fuse)** for cross-modal semantic understanding, linking visual gestures to their intended textual commands.

### Key Components

- **Feature Extraction with a Vision Transformer (ViT):**

  - Instead of relying only on landmark coordinates, we will use a pre-trained Vision Transformer (ViT) to extract rich features from the entire hand image. The ViT's ability to capture global context in an image makes it ideal for understanding the nuances of hand shapes and orientations.
  - We'll use the bounding box of the detected hand (from MediaPipe) to crop the hand image and feed it into the ViT.

- **Gesture Classification:**

  - **Static Gestures:** Fine-tune the ViT model to classify static gestures based on the rich features it extracts. This should be more robust and accurate than the simple MLP.
  - **Dynamic Gestures:** Feed the sequence of feature vectors from the ViT (one for each frame) into an LSTM. This approach, inspired by the HAVIT paper's methods for video classification, will provide the LSTM with much more contextual information than just a single point's history, enabling the recognition of more complex dynamic gestures.

- **Semantic Understanding (ALBEF-inspired):**
  - To bridge the gap between gesture recognition and action, we will implement a text-gesture alignment model inspired by the ALBEF (Align Before Fuse) framework.
  - **Training:** We'll use a contrastive learning approach to train a multimodal model. The model will learn to create similar vector representations for a gesture's visual features (from the ViT) and the text description of its corresponding command (from a text encoder like BERT).
  - **Inference:** When a gesture is performed, the system will find the text command whose vector representation is closest to the gesture's vector representation, effectively "understanding" the semantic meaning of the gesture (e.g., a "thumbs up" gesture maps to the "like" command).

### Benefits of the Advanced Approach

- **Higher Accuracy:** ViTs are state-of-the-art in many computer vision tasks and can capture more complex features than simple landmark-based models.
- **Improved Robustness:** By analyzing the whole hand image, the ViT-based approach is expected to be more robust to variations in lighting, hand size, and orientation.
- **Semantic Understanding:** The ALBEF-inspired model will allow for a more flexible and powerful way to map gestures to actions, moving beyond a hardcoded mapping to a learned association between what a gesture _looks like_ and what it _means_.
- **Extensibility:** This architecture is more extensible. Adding new gestures and commands would involve retraining the model with new data, rather than writing new code for gesture-action mapping.

## References

- [HAVIT: research on vision-language gesture interaction mechanism for smart furniture](https://www.nature.com/articles/s41598-025-10758-9)
- [Hand gesture recognition with mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)
- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
