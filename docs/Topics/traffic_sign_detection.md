# Traffic Sign Detection

Traffic signs are pivotal components of traffic management, ensuring the regulation and safety of road traffic. However, existing detection methods suffer from **poor accuracy** and **poor real-time performance** in dynamic road environments.
Traffic sign detection utilizes _computer vision_ and _artificial intelligence_ technologies to recognize and interpret traffic signs on the road automatically. This technology is critical for enhancing:

- driving safety
- guiding traffic behavior
- aiding in the decision-making processes of autonomous vehicles

## Importance of Effective Traffic Sign Detection

- Helps drivers adhere to traffic regulations
- Reduces traffic accidents
- Improves traffic flow management
- Ensure road safety

## Challenges faced in Traffic Sign Detection

- Growing variety and complexity of urban traffic signs
- Dynamic and complex traffic conditions
- Lighting variations
- Occlusions
- Similarities among different types of signs

> [!NOTE]
> Deep-learning based traffic sign recognition algorithms have prominence in helping overcoming those challenges.

## Related Concepts

- Principles of small object detection layers
- YOLOv12
- BiFPN network

## Essential Key Benefits

- Vehicle-mounted traffic sign recognition systems:
  - Effectively guide and regulate driver behavior
  - Ensure safe driving
  - Reduce traffic accidents
- Detecting and recognizing traffic signs:
  - Further enhances autonomous driving techonology
  - Improves road commuting efficiency

### Main Steps of Traffic Sign Detection

- Localization
  - Identification of the **presence** and **exact location** of traffic signs within complex road environments.
- Recognition and classification of signs
  - Interprets the detected signs to determine their **types**, such as _stops signs_ and _speed limits_.

## Traditional Methods and Related Works

- Image processing and machine learning
- Detection algorithms:
  - Histogram of Orientation Gradients (HOG)
  - Scale-Invariant Feature Transformations (SIFT)
- Classification algorithms:
  - Support Vector Machines (SVM)
  - Random Forest (RF)

### Traditional Approach

- Relies on _distinctive features_ of traffic signs
  - Specific colors
  - Prominent shapes
- Extraction of those features for detection
- Classification with a **_trained classifier_**

#### Related Researches

- ["Road traffic sign detection and classification" De La Escalera et al.](https://ieeexplore.ieee.org/document/649946)
  - Traffic signs extraction from their environment using the color and shape of the sign's corners along with classification of identified signs using neural networks.
- ["Goal evaluation of segmentation algorithm for traffic sign recognition" by Gomez-Moreno et al.](https://ieeexplore.ieee.org/document/4220659)
  - Color segmentation method using SVM and speed increase using Look-Up Table (LUT)
- ["Robust traffic sign recognition based on color global and Local Oriented Edge Magnitude Patterns (LOEMP)" by Yuan et al.](https://ieeexplore.ieee.org/document/6810168)
  - Integrates:
    - color
    - global spatial structure
    - local shape information

  which significantly increasing the efficiency of sign detection.

- ["On circular traffic sign detection and recognition" by Berkaya et al.](https://www.sciencedirect.com/science/article/abs/pii/S0957417415007848)
  - EDCircles circle detection algorithm combined with an RGB-based color thresholding technique to detect traffic signs
  - Integrates GABOR, LBP, and HOG techniques
  - Classification was performed with an SVM
  - Enhanced traffic sign detection process

> [!IMPORTANT]
> Color features and shape features are prone to interference from external objects of similar color and shape in _complex road conditions_, resulting in **poor generalization**.

