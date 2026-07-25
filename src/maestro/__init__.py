"""Maestro — real-time hand gesture recognition for presentation control."""

import os

# Prevent macOS OMP Error #15 when multiple OpenMP runtimes (e.g. from torch and numpy) are loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = ["app", "domain", "interfaces", "pipeline", "training", "evaluation", "utils"]
