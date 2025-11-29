# Hand Gesture Recognition System

This project is a comprehensive exploration into the field of **Human-Computer
Interaction (HCI)**, focusing on the development of a sophisticated **Hand
Gesture Recognition System**. By leveraging cutting-edge techniques in
**Computer Vision** and **Machine Learning**, we aim to create a system that
can accurately interpret and translate human hand gestures into actionable
commands, providing a seamless and intuitive way to interact with computing
devices.

## Introduction

The primary goal of this project is to design and implement a robust gesture
recognition system that goes beyond simple pattern matching. We are building
a system that understands the semantic meaning behind gestures, enabling a more
natural and powerful user experience. Our approach is inspired by
state-of-the-art research in the field, including the **HAVIT** (Hierarchical
Attention-based Vision Transformer) and **ALBEF** (Align before Fuse) models.

The system architecture is composed of four main components:

- **Input Manager**: Responsible for capturing video frames from a camera
  and preprocessing them for the recognition engine.
- **Recognition Engine**: The core of the system, which uses a Vision
  Transformer (ViT) based model to analyze the visual data and recognize the
  performed gestures.
- **Action Mapper**: Translates the recognized gestures into specific
  commands or actions.
- **OS Controller**: Executes the commands on the underlying operating system.

A key feature of our project is the focus on **zero-shot learning**, which will
allow the system to recognize new gestures without the need for retraining. This
is achieved through a cross-modal learning approach that bridges the gap between
vision and language, enabling the system to understand and generalize
gesture-based commands.

This repository contains all the code, documentation, and research materials
related to the project.

## Installation

To set up the project for development:

```bash
# Install the project in editable mode
pip install -e .
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Quick Start

After installation, you can use the input manager:

```python
from input_manager import InputManager

manager = InputManager(camera_index=0)
manager.start_capture()
frame_data = manager.capture_frame()
manager.stop_capture()
```

See `src/input_manager/example_usage.py` for more examples.

## Group Members

- Sley Hortes
- Neil Taison Rigaud
- Khishig Tuguldur
- Eder Guerrero
