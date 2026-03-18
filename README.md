# Presentation and Meeting Control System

Production-grade foundation for a webcam-based gesture control system using MediaPipe Hands and LSTM.

## Phase 1 Scope

This repository currently includes:

- Project scaffolding and package layout
- Interface contracts for core modules
- Typed configuration schema and loader
- CI skeleton for lint and unit tests

## Quick Start

1. Create or activate a Python 3.11+ environment.
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
