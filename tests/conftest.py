from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Collection guard: skip training tests when scikit-learn is not installed.
# pytest imports every collected file to register marks; files that pull in
# sklearn at module-level therefore fail collection, not just execution, when
# the package is absent.  `collect_ignore` prevents the import entirely.
# ---------------------------------------------------------------------------
_TRAINING_TESTS = [
    "unit/test_training_datamodule.py",
    "unit/test_training_export.py",
    "integration/test_training_pipeline_smoke.py",
]

_HERE = Path(__file__).parent

collect_ignore: list[str] = []
try:
    _sklearn_present = importlib.util.find_spec("sklearn") is not None
except (ModuleNotFoundError, ValueError):
    _sklearn_present = False

if not _sklearn_present:
    collect_ignore.extend(str(_HERE / p) for p in _TRAINING_TESTS)


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"
