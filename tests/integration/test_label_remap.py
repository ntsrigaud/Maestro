"""Integration tests: label_remap parsing and runtime application.

Verifies that:
  1. ``_load_command_map`` correctly parses the ``label_remap`` section from YAML.
  2. ``build_container`` exposes the remap on ``AppContainer.label_remap``.
  3. The remapping produces the expected command_id when a model predicts the
     "wrong" swipe direction (e.g. swiping_left → dispatches next_slide because
     swiping_left is remapped to swiping_right in the production command map).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from maestro.app.bootstrap import _load_command_map, build_container
from maestro.domain.entities import StabilizedPrediction
from maestro.infrastructure.model.checkpoint_loader import InferenceArtifact
from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM
from maestro.pipeline.activation import GestureActivationController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_command_map(tmp_path: Path, *, include_remap: bool = True) -> Path:
    lines = [
        "version: test_v1",
        "gesture_commands:",
        "  swiping_right: next_slide",
        "  swiping_left: previous_slide",
        "  unknown: no_action",
    ]
    if include_remap:
        lines += [
            "label_remap:",
            "  swiping_left: swiping_right",
            "  swiping_right: swiping_left",
        ]
    path = tmp_path / "command_map.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _pred(label: str, accepted: bool = True) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=0.95, accepted=accepted)


# ---------------------------------------------------------------------------
# _load_command_map — YAML parsing
# ---------------------------------------------------------------------------


def test_load_command_map_returns_label_remap(tmp_path: Path) -> None:
    """`_load_command_map` must return the label_remap dict when present in YAML."""
    path = _write_command_map(tmp_path, include_remap=True)
    command_map, continuous_labels, label_remap = _load_command_map(path)

    assert label_remap == {"swiping_left": "swiping_right", "swiping_right": "swiping_left"}


def test_load_command_map_empty_remap_when_section_absent(tmp_path: Path) -> None:
    """`_load_command_map` must return an empty dict when label_remap is absent."""
    path = _write_command_map(tmp_path, include_remap=False)
    _, _, label_remap = _load_command_map(path)
    assert label_remap == {}


def test_production_command_map_has_swipe_remap(config_dir: Path) -> None:
    """The production command_map_presentation.yaml must contain the swipe remap."""
    path = config_dir / "gestures" / "command_map_presentation.yaml"
    _, _, label_remap = _load_command_map(path)
    assert label_remap.get("swiping_left") == "swiping_right", (
        "swiping_left must remap to swiping_right in production command map"
    )
    assert label_remap.get("swiping_right") == "swiping_left", (
        "swiping_right must remap to swiping_left in production command map"
    )


# ---------------------------------------------------------------------------
# build_container — label_remap is wired into AppContainer
# ---------------------------------------------------------------------------


def _mock_artifact() -> InferenceArtifact:
    model = EnhancedTwoStreamLSTM(
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=10,
        num_heads=4,
        proj_dim=64,
    )
    return InferenceArtifact(
        model=model,
        model_version="test_remap_v1",
        class_labels=[
            "unknown",
            "point_one",
            "point_two",
            "swiping_up",
            "swiping_down",
            "swiping_left",
            "swiping_right",
            "fist",
            "zooming_in_full_hand",
            "zooming_out_full_hand",
        ],
        feature_schema_version="feature-schema-v5",
        feature_dim=147,
        window_length=32,
        window_step=3,
        orientation_normalization=False,
        artifact_path=Path("/dev/null"),
    )


@patch("maestro.app.bootstrap._load_model_from_source")
@patch("maestro.app.bootstrap.validate_runtime_compatibility")
@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
@patch("maestro.app.bootstrap.MediaPipeHandsExtractor")
def test_container_exposes_label_remap(
    _mock_extractor: MagicMock,
    _mock_capture: MagicMock,
    _mock_validate: MagicMock,
    mock_load: MagicMock,
    config_dir: Path,
) -> None:
    """``build_container`` must populate ``AppContainer.label_remap`` from YAML."""
    mock_load.return_value = _mock_artifact()
    container = build_container(str(config_dir))
    assert container.label_remap.get("swiping_left") == "swiping_right"
    assert container.label_remap.get("swiping_right") == "swiping_left"


# ---------------------------------------------------------------------------
# Remapping semantics: model swiping_left → dispatches next_slide
# ---------------------------------------------------------------------------


def test_remap_swiping_left_dispatches_next_slide() -> None:
    """After remapping, a model prediction of swiping_left must dispatch next_slide.

    The production command_map has swiping_right → next_slide.  Because
    swiping_left is remapped to swiping_right before routing, the activation
    controller sees swiping_right and emits the next_slide command.
    """
    label_remap = {"swiping_left": "swiping_right", "swiping_right": "swiping_left"}
    command_map = {"swiping_right": "next_slide", "swiping_left": "previous_slide"}
    ctrl = GestureActivationController(
        command_map=command_map,
        command_cooldown_ms=50,
    )

    # Simulate: model predicts swiping_left → apply remap → route.
    raw_pred = _pred("swiping_left")
    remapped_label = label_remap.get(raw_pred.label, raw_pred.label)
    routed_pred = StabilizedPrediction(
        label=remapped_label,
        confidence=raw_pred.confidence,
        accepted=raw_pred.accepted,
        reason=raw_pred.reason,
    )
    cmd = ctrl.route(routed_pred)

    assert cmd is not None
    assert cmd.command_id == "next_slide", (
        f"Expected next_slide after remap, got {cmd.command_id!r}"
    )
    assert cmd.source_gesture == "swiping_right", (
        "source_gesture must reflect the remapped label, not the raw model output"
    )


def test_remap_swiping_right_dispatches_previous_slide() -> None:
    """The reverse: model prediction of swiping_right → dispatches previous_slide."""
    label_remap = {"swiping_left": "swiping_right", "swiping_right": "swiping_left"}
    command_map = {"swiping_right": "next_slide", "swiping_left": "previous_slide"}
    ctrl = GestureActivationController(
        command_map=command_map,
        command_cooldown_ms=50,
    )

    raw_pred = _pred("swiping_right")
    remapped_label = label_remap.get(raw_pred.label, raw_pred.label)
    routed_pred = StabilizedPrediction(
        label=remapped_label,
        confidence=raw_pred.confidence,
        accepted=raw_pred.accepted,
        reason=raw_pred.reason,
    )
    cmd = ctrl.route(routed_pred)

    assert cmd is not None
    assert cmd.command_id == "previous_slide"
    assert cmd.source_gesture == "swiping_left"


def test_non_swipe_labels_not_affected_by_remap() -> None:
    """Labels not listed in label_remap must pass through unchanged."""
    label_remap = {"swiping_left": "swiping_right", "swiping_right": "swiping_left"}
    command_map = {
        "swiping_right": "next_slide",
        "swiping_left": "previous_slide",
        "fist": "erase_annotations",
    }
    ctrl = GestureActivationController(
        command_map=command_map,
        command_cooldown_ms=50,
    )

    raw_pred = _pred("fist")
    remapped_label = label_remap.get(raw_pred.label, raw_pred.label)
    assert remapped_label == "fist", "fist must not be affected by swipe remap"

    routed_pred = StabilizedPrediction(
        label=remapped_label,
        confidence=raw_pred.confidence,
        accepted=raw_pred.accepted,
        reason=raw_pred.reason,
    )
    cmd = ctrl.route(routed_pred)
    assert cmd is not None
    assert cmd.command_id == "erase_annotations"
    assert cmd.source_gesture == "fist"
