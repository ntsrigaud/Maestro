# Presentation and Meeting Control System

Production-grade foundation for a webcam-based gesture control system using MediaPipe Hands and LSTM.

## Current Scope

This repository currently includes:

- Phase 1: project scaffolding, package layout, interface contracts, typed config schema, and CI quality gates.
- Phase 2: OpenCV frame source, MediaPipe landmark extractor integration, bootstrap/runtime wiring, and validated unit/integration tests.

## Phase 2 Validation Snapshot (2026-03-18)

- `ruff check src/gesture_control/ tests/` passed.
- `pytest -v` passed with 12 tests.
- `mypy src/gesture_control/infrastructure/` passed.
- Bootstrap/runtime wiring validated in Python 3.11 target configuration.

## Quick Start

1. Create or activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements/dev.txt
```

3. Run quality checks:

```bash
ruff check .
pytest
```

## Repository Design

See `ARCHITECTURE_SPECIFICATION.md` for the complete target architecture and implementation roadmap.
