# Recognition Engine Module

This document outlines the technical design for the Recognition Engine, the core component responsible for identifying hand gestures from video frames.

## 1. Purpose

The Recognition Engine receives an image frame from the Input Manager, analyzes it to detect and classify a user's hand gesture, and outputs a standardized gesture label (e.g., `FIST`, `POINTER`) to the Action Mapper module.

## 2. Core Technology & Methodology

The architecture is inspired by the **HAVIT research paper**, which demonstrates the effectiveness of a **Vision Transformer (ViT)** for gesture recognition. While the paper also explores vision-language fusion with ALBEF, for the purpose of direct mouse control, we will focus on implementing a powerful ViT-based image classifier.

The process will be a **two-stage pipeline**:
1.  **Hand Detection:** First, locate the hand in the image.
2.  **Gesture Classification:** Second, classify the gesture of the located hand.

## 3. Requirements & Dependencies

-   **Deep Learning Framework:** **PyTorch** will be used for model implementation and training.
-   **Model Library:** **`timm` (PyTorch Image Models)** is essential for accessing pre-trained Vision Transformer models.
-   **Object Detection:** A library or pre-trained model for hand detection (e.g., a YOLO model from Ultralytics).
-   **Image Processing:** **OpenCV (`opencv-python`)** for image manipulation (cropping, resizing, normalization).

## 4. Implementation Plan

### Stage 1: Hand Detection

Before classifying a gesture, the hand must be accurately located within the frame. We will use a pre-trained **YOLO (You Only Look Once)** model for this task. YOLO is a fast and accurate object detection algorithm, making it suitable for real-time applications.

-   **Action:** The YOLO model will process the input frame and output bounding box coordinates for any hands it detects.

### Stage 2: Gesture Classification with Vision Transformer (ViT)

Once the hand is located, we will use a ViT to classify its gesture.

1.  **Image Preprocessing:**
    -   The image frame is cropped using the bounding box from the YOLO model.
    -   The cropped hand image is resized to the ViT's required input dimensions (e.g., 224x224 pixels).
    -   The image is normalized using the standard mean and standard deviation values for ImageNet, as the ViT is pre-trained on it.

2.  **Model Inference:**
    -   The preprocessed image tensor is fed into the fine-tuned Vision Transformer model.
    -   The model performs a forward pass and outputs a set of probabilities for each possible gesture class.

3.  **Output:**
    -   The gesture with the highest probability is selected.
    -   The module returns a string label for the recognized gesture (e.g., `"POINTING_FINGER"`).

## 5. Model Training and Data

-   **Pre-trained Model:** We will not train a ViT from scratch. Instead, we will use a model pre-trained on the ImageNet dataset, such as `vit_base_patch16_224` available through the `timm` library.
-   **Fine-Tuning:** The crucial step is to **fine-tune** this pre-trained model on a specific hand gesture dataset. This adapts the model to our specific task.
-   **Dataset:** The **HaGRID (Hand Gesture Recognition Image Dataset)**, mentioned in the HAVIT paper, is the ideal candidate. We will need to select a subset of its 18 gestures that are intuitive for mouse control, such as:
    -   `fist` (for clicking/dragging)
    -   `pointing_up` or `index` (for cursor movement)
    -   `open_palm` or `five` (for releasing/idle)
    -   `two` (for right-click)
    -   `three` (for middle-click)

This two-stage approach separates the concerns of detection and classification, allowing us to use the best available models for each task while building on the state-of-the-art methods identified in our research.