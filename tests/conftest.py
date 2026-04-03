from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"
