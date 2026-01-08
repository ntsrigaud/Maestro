# GesTop: Customizable Gesture Control of Computer Systems

## Key Points

- Configuration for remapping gestures and actions
  - Allow users to add new gestures
- Static and dynamic gestures
- Dynamic gestures
  - Sequence of poses
  - Combination of hand motion along with continous pose

---

## Architecture

### Gesture Receiver

- Receives keypoints from the image and passes it to the **Gesture Recognizer**.

### Gesture Recognizer

- Uses neural network to classify both static and dynamic gestures.

### Gesture Executor

- Executes an action based on the detected gesture.

---

## Additional Notes

- **Customizable system**
  - Map gesturess to arbitrary desktop actions including _shell scripts_.
  - More flexible
- Use **Mediapipe** to detect **_hand keypoints_** for individual video frames.
- **Keypoints** are used to detect static and dynamic gestures using their neural networks.

### GestARlite

- Lightweight, on device framework for detecting gestures based on _pointing fingers_.
- **Challenge**:
  - Find a method for accurate gesture recognition
  - **Pre-existing Methods**:
    - SOA deep learning models -> Computationally expensive
    - Expensive sensors
  - **Solution**:
    - Lightweight hand gesture recognition framework
      - Works in _First Person View_ for wearable devices
      - **FPV** -> Study and development of CV techniques considering images and videos acquired from a camera mounted on the head of a person.
    - Models trained on GPU machines and ported on android smartphone to be used with wearable devices.
    - **[MobileNet V2](https://arxiv.org/abs/1801.04381)** for hand localization
    - Custom fingertip regression
      - Precisely track and locate human fingertips within digital images and video streams.
      - Predict their exact coordinates in a continuous manner.
    - Bi-LSTM for gesture classification
  - Works well in real-time on mobile devices.
    - Achieves 80% classification accuracy on [EgoGestAR pointing gesture video dataset](https://github.com/varunj/EgoGestAR)
      - Average latency of 0.12s

---

## Gesture Recognition Platforms

- Controlling virtual interfaces
- Gaming
- Embodied AR/VR environments
- Automotive Human-machine interaction
- Home automation
- Education
- Retail business environments
- Consumer electronics controls

---

## Open Source Gesture Recognition Platforms

**[GRT](https://github.com/nickgillian/grt)**: C++ machine learning library for real-time gesture recognition

- Used for **Gesture Recognizer**
- _Can we obtain better real-time performance from this?_

**Gesture Coder** learns from multi-touch gesture examples and generates imperative recognition code automatically.

- Invoke corresponding application actions.
- GesTop uses deep learning models.

---

## Design Architecture

- Mediapipe
  - Detect hand keypoints and pass them to **Gesture Receiver**
- **Gesture Receiver**
  - Passes keypoints to **Gesture Recognizer**
- **Gesture Recognizer**
  - Processes keypoints into encoded features.
  - Encoded features -> Neural network - > Detect output gesture
- **Gesture Receiver**
  - Sends detected gesture to **Gesture Recognizer**
- **Gesture Executor**
  - Executes the **_mapped_** action.

### Additional Notes

- **Mouse tracker**
  - Tracks the cursor on the screen as the hand moves.
  - Uses the tip of the index finger as keypoint.
- **Gesture recognizer** classifies gestures _using keypoints_
  - Two neural networks:
    - One for static
    - One for dynamic
- **Gesture executor** uses gesture name to find the **mapped** action and executes it.
  - Lookup
- They recorded their own data and created a small dataset.
  - Around 2000 samples in a couple of minutes.
  - Not much public dataset availability.
- Trained dynamic gestures on [SHREC 2021 dataset](https://arxiv.org/abs/2106.10980)
- New actions and gestures can be added **_at will_**.
  - Flexible system
  - Can be done using script used to collect initial data for training.
  - **Retrain the network to detect new gestures**
    - _Can it be done using zero-shot learning?_

---

## Implementation and Results

### Static Gestures

- Used FFN with 2 linear layers
  - Trained around 50 epochs
- Introduced a `NONE` gesture when no relevant gesture is detected.
  - Captured unrelated static gestures and labelled them as `NONE`
- **High detection accuracy** for multiple users and **different lighting conditions**
  - Validation accuracy -> 99.12 %
  - **No noticeable latency**

### Dynamic Gestures

- RNN using a linear layer to encode incoming features
- Bidirection GRU
- **Key issue**:
  - Compute start and end of gesture precisely.
    - Uses _signal key_ (`CTRL`) to signify start and end of a gesture.
  - Lower performance than static gestures
    - 85% average accuracy
    - _Pose_ of the hand and _displacement_ over time
- Gestures which involves **displacement** of the hand are detected well, whereas those concerned with the **orientation** of the hand have lower accuracy.

### Additional Notes

- **Domain Mismatch Problem**
  - SHREC dataset was filmed using higher resolution _depth_ camera (Intel RealSense)
  - Incoming streams is from RGB camera
  - Loss in accuracy during training

---

## Ongoing Work

- Improve feature computation
- Training on a larger, diverse dataset

## Conclusion

- GesTop control desktop through hand gestures
- Can be customized
- **_Fully functional replacement for the mouse_**
- Easy to **extend** by adding new gestures and actions
  - Many more desktop use cases.

## Future Work

- Improving detection accuracy
- Make incremental training for new gestures efficient
  - Zero-shot learning ???
- Detecting gesture _start/end_ and other user subtle contents
- Conduct user studies to measure usability of the tool
