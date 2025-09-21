# Project Architecture

This document outlines the software architecture for the Hand Gesture Recognition project. The system is designed as a modular pipeline, with each component responsible for a specific part of the gesture recognition and control process.

## System Overview

The system consists of four main modules:

1.  **Input Manager:** Captures video from the camera.
2.  **Recognition Engine:** Performs hand gesture recognition.
3.  **Action Mapper:** Translates gestures into commands.
4.  **OS Controller:** Executes commands on the host OS.

```
+-------------------+      +----------------------+      +---------------------+      +--------------------+
| 1. Input Manager  |----->| 2. Recognition       |----->| 3. Action Mapper    |----->| 4. OS Controller   |
| (Webcam Capture)  |      |    Engine (AI/ML)    |      | (Gesture-to-Action) |      | (Virtual Mouse)    |
+-------------------+      +----------------------+      +---------------------+      +--------------------+
```

## Technical Direction

The core of this project is the **Recognition Engine**, which will be developed using a state-of-the-art deep learning approach. Our research has shown that traditional methods relying solely on hand landmark coordinates can be limited. Therefore, we will be leveraging a **Vision Transformer (ViT)** based architecture for more robust and accurate gesture recognition.

Furthermore, to create a truly intuitive and extensible system, we will be implementing a semantic understanding layer inspired by the **ALBEF (Align Before Fuse)** framework. This will allow the system to learn the _meaning_ of gestures by associating them with natural language descriptions of commands, moving beyond hardcoded gesture-to-action mappings.

This approach offers several key benefits:

- **Higher Accuracy and Robustness:** By analyzing the entire hand image, the ViT can capture more nuanced features than landmark-based methods alone.
- **Semantic Understanding:** The system will be able to understand what gestures _mean_, making it more flexible and easier to customize.
- **Extensibility:** Adding new gestures and commands will be a matter of providing new training data, not rewriting code.

For more detailed information on the technical design and implementation plan for each module, please refer to the `README.md` file within each module's directory.
