# Tests — Architecture A

Unit and integration tests for the dual-stream supervised gesture recognition system.

## Structure

```
tests/
    test_hand_detection.py        MediaPipe hand detector + HandDetectionResult
    test_hand_cropper.py          Bounding-box crop + ImageNet normalisation
    test_preprocessing.py         center_crop_frame pipeline
    test_gcn_stream.py            GCN geometry stream forward pass  (Phase 1)
    test_rgb_stream.py            ViT-B/16 appearance stream forward pass  (Phase 2)
    test_dual_stream.py           Dual-stream fusion model forward pass  (Phase 3+4)
    test_dataset.py               HaGRID dataset loading and train/val/test splits
    test_keypoint_extraction.py   Offline MediaPipe keypoint extraction
```

## Running Tests

```bash
make test          # full suite (verbose)
make test-fast     # stop on first failure
```

Or directly:

```bash
pytest tests/ -v
pytest tests/test_hand_detection.py -v   # single file
```

## Test Philosophy

- **Unit tests** validate individual components in isolation with synthetic / mocked data.
- **Integration tests** validate end-to-end pipeline behaviour.
- All tests must pass before initiating training (`make train-gcn`).
- Tests tagged `@pytest.mark.slow` require real model weights and are skipped in CI by default.

## Writing New Tests

```python
import pytest
import torch

@pytest.fixture
def dummy_keypoints():
    """21 keypoints × 3 coordinates (x, y, z in [0,1])."""
    return torch.rand(21, 3)

def test_gcn_forward(dummy_keypoints):
    from semantic_hgr.models.gcn_stream import GCNStream
    model = GCNStream(in_channels=3, hidden_channels=128, num_classes=8)
    out = model(dummy_keypoints.unsqueeze(0))
    assert out.shape == (1, 8)
```
