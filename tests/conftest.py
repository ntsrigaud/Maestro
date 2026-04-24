from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Collection guard: skip training tests when scikit-learn is not installed.
#
# pytest imports every collected file to register marks; files that import
# training modules (which transitively import sklearn) fail at collection time,
# not execution time, so `pytestmark` alone is insufficient.
#
# Convention: any test file whose name contains "training" is considered a
# training test and requires the [training] extra.  New training tests are
# covered automatically as long as they follow this naming convention —
# no list to maintain.
# ---------------------------------------------------------------------------
try:
    _sklearn_present = importlib.util.find_spec("sklearn") is not None
except (ModuleNotFoundError, ValueError):
    _sklearn_present = False

try:
    _pyside6_present = importlib.util.find_spec("PySide6") is not None
except (ModuleNotFoundError, ValueError):
    _pyside6_present = False


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Prevent collection of training tests when scikit-learn is absent,
    and UI tests when PySide6 is absent."""
    if not _sklearn_present and "training" in collection_path.name:
        return True
    if not _pyside6_present and collection_path.name.startswith("test_ui_"):
        return True
    return None


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"
