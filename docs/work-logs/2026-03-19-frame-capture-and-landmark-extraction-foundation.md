# Work Log - Frame Capture and Landmark Extraction Foundation

## Context
This work implements the Phase 2 runtime ingestion foundation by integrating OpenCV-based frame capture and MediaPipe Hands landmark extraction into the application runtime flow, with dependency updates and test coverage to validate production-path behavior.

## Scope of Work
- Initialized infrastructure package boundaries for camera and MediaPipe integrations.
- Implemented concrete frame source and landmark extractor modules.
- Updated interface contracts and exports to support extractor lifecycle management.
- Wired camera source and landmark extractor into application bootstrap and runtime loop.
- Added dependency updates for OpenCV and MediaPipe, with Python version pinning aligned to compatibility constraints.
- Added unit and integration tests for ingestion pipeline behavior.
- Updated project scope documentation to reflect completed ingestion capabilities.

## Technical Changes
- Added infrastructure modules:
- `src/gesture_control/infrastructure/camera/opencv_source.py` for frame acquisition, FPS pacing, frame ID sequencing, and camera release handling.
- `src/gesture_control/infrastructure/mediapipe/hands_extractor.py` for RGB conversion, hand landmark extraction, confidence evaluation, and validity gating.
- Added package initializers in `src/gesture_control/infrastructure/`, `src/gesture_control/infrastructure/camera/`, and `src/gesture_control/infrastructure/mediapipe/`.
- Updated `src/gesture_control/interfaces/landmark_extractor.py` to include extractor lifecycle closure contract.
- Updated `src/gesture_control/interfaces/__init__.py` to export protocol interfaces through a centralized surface.
- Updated `src/gesture_control/app/bootstrap.py` to compose `OpenCVFrameSource` and `MediaPipeHandsExtractor` into `AppContainer`.
- Updated `src/gesture_control/app/runtime.py` to execute frame-to-landmark ingestion loop with diagnostic counters and graceful shutdown of source/extractor resources.
- Updated dependency configuration in `requirements/base.txt` and `pyproject.toml` with `opencv-python-headless`, `mediapipe`, and Python `>=3.11,<3.12` constraints.
- Updated `README.md` current capability summary and removed completed Phase 1/2 reference links.

## Testing and Validation
- Added unit tests in `tests/unit/test_opencv_source.py` for camera initialization, frame read behavior, frame ID progression, capture failure handling, and close semantics.
- Added unit tests in `tests/unit/test_mediapipe_extractor.py` for no-hand behavior, valid hand extraction, confidence rejection, and extractor close semantics.
- Added integration tests in `tests/integration/test_phase2_pipeline.py` for frame-to-landmark ingestion behavior and no-hand pipeline handling.
- Executed test suite; `pytest -q` passed.
- No test suppressions were added or removed.
- Build/dependency updates were validated against Python 3.11 compatibility requirements.

## Outcome
The branch delivers an operational ingestion baseline that captures webcam frames and produces confidence-gated landmark packets through the runtime pipeline, with deterministic tests and explicit resource lifecycle handling. This extends runtime functionality and infrastructure integration without introducing breaking API changes.
