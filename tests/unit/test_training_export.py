from __future__ import annotations

from pathlib import Path

import pytest

from maestro.training.export_checkpoint import compute_sha256

pytestmark = pytest.mark.training


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    digest = compute_sha256(path)
    assert len(digest) == 64
