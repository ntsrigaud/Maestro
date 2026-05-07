from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from maestro.utils.schema_validation import (
    AppConfig,
    InferenceConfig,
    RuntimeConfig,
)

_EXCLUDE_FIELDS = frozenset({"command_map", "app_name"})


class ConfigPanel(QWidget):
    """
    Form-based editor for AppConfig and InferenceConfig.

    Signals
    -------
    config_changed(bool) — True when the in-memory draft is valid.
    apply_requested      — User clicked "Apply & Restart".
    """

    config_changed: Signal = Signal(bool)  # True = valid draft
    apply_requested: Signal = Signal()

    def __init__(
        self,
        config_path: str = "configs",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._widgets: dict[str, QWidget] = {}  # field_name → widget
        self._valid = True
        self._runtime_active = False
        self._base_app_vals: dict[str, Any] = {}
        self._base_infer_vals: dict[str, Any] = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vbox = QVBoxLayout(inner)

        vbox.addWidget(self._build_group("App", AppConfig))
        vbox.addWidget(self._build_group("Inference", InferenceConfig))
        vbox.addWidget(self._build_gaze_group())

        self._apply_btn = QPushButton("Apply && Restart")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        vbox.addWidget(self._apply_btn)
        vbox.addStretch()

        scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        self._load_from_yaml()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_runtime_active(self, active: bool) -> None:
        """Disable Apply when the runtime is running."""
        self._runtime_active = active
        self._refresh_apply_button()

    def current_config(self) -> RuntimeConfig | None:
        """Return the validated draft, or None if currently invalid."""
        if not self._valid:
            return None
        try:
            return self._build_runtime_config()
        except (ValidationError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Private — form construction
    # ------------------------------------------------------------------

    def _build_group(self, title: str, model_cls: type[BaseModel]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for name, info in model_cls.model_fields.items():
            if name in _EXCLUDE_FIELDS:
                continue
            widget = self._make_widget(name, info)
            if widget is not None:
                self._widgets[name] = widget
                form.addRow(name.replace("_", " ").title() + ":", widget)
        return box

    def _build_gaze_group(self) -> QGroupBox:
        """Build a dedicated group box exposing ``gaze.enabled``.

        The remaining GazeConfig fields (thresholds, timing) are advanced
        parameters that most users never need to change; they remain
        YAML-only and are preserved transparently when the config is saved.
        """
        box = QGroupBox("Gaze Detection")
        form = QFormLayout(box)
        cb = QCheckBox()
        cb.setToolTip(
            "When enabled, gesture commands are only dispatched while the user\n"
            "is looking at the camera (head-pose gate). Disable to accept gestures\n"
            "regardless of head orientation."
        )
        cb.stateChanged.connect(self._on_any_change)
        self._widgets["gaze_enabled"] = cb
        form.addRow("Enable gaze gate:", cb)
        return box

    def _make_widget(self, name: str, info: Any) -> QWidget | None:
        annotation = info.annotation
        # bool must come before int (bool is subclass of int in Python)
        if annotation is bool:
            check = QCheckBox()
            check.stateChanged.connect(self._on_any_change)
            return check
        if annotation is int:
            spin = QSpinBox()
            ge_val, le_val = _ge(info), _le(info)
            spin.setMinimum(int(ge_val) if ge_val is not None else -999_999)
            spin.setMaximum(int(le_val) if le_val is not None else 999_999)
            spin.valueChanged.connect(self._on_any_change)
            return spin
        if annotation is float:
            dspin = QDoubleSpinBox()
            ge_val, le_val = _ge(info), _le(info)
            dspin.setMinimum(ge_val if ge_val is not None else -1e9)
            dspin.setMaximum(le_val if le_val is not None else 1e9)
            dspin.setSingleStep(0.05)
            dspin.valueChanged.connect(self._on_any_change)
            return dspin
        if annotation is str:
            edit = QLineEdit()
            edit.textChanged.connect(self._on_any_change)
            return edit
        # Literal — use QComboBox with literal choices
        if typing.get_origin(annotation) is typing.Literal:
            choices = annotation.__args__
            combo = QComboBox()
            for c in choices:
                combo.addItem(str(c))
            combo.currentIndexChanged.connect(self._on_any_change)
            return combo
        return None  # unsupported / nested type; skip field

    # ------------------------------------------------------------------
    # Private — validation
    # ------------------------------------------------------------------

    def _on_any_change(self, *_args: Any) -> None:
        try:
            self._build_runtime_config()
            self._valid = True
        except (ValidationError, ValueError, AttributeError):
            self._valid = False
        self.config_changed.emit(self._valid)
        self._refresh_apply_button()

    def _build_runtime_config(self) -> RuntimeConfig:
        """Collect widget values and attempt Pydantic validation."""
        values: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        # gaze_enabled maps to the nested gaze.enabled field — handle separately
        gaze_enabled = values.pop("gaze_enabled", None)
        # Partition into AppConfig / InferenceConfig fields
        app_fields = set(AppConfig.model_fields) - _EXCLUDE_FIELDS
        infer_fields = set(InferenceConfig.model_fields)
        app_vals = {k: v for k, v in values.items() if k in app_fields}
        infer_vals = {k: v for k, v in values.items() if k in infer_fields}
        # Build gaze sub-config: preserve YAML-only fields, only override enabled
        base_gaze: dict[str, Any] = dict(self._base_infer_vals.get("gaze", {}))
        if gaze_enabled is not None:
            base_gaze["enabled"] = gaze_enabled
        infer_vals["gaze"] = base_gaze
        # Preserve non-editable fields from the loaded config
        return RuntimeConfig(
            app=AppConfig(**{**self._base_app_vals, **app_vals}),
            inference=InferenceConfig(**{**self._base_infer_vals, **infer_vals}),
        )

    def _refresh_apply_button(self) -> None:
        self._apply_btn.setEnabled(self._valid and not self._runtime_active)

    # ------------------------------------------------------------------
    # Private — load / apply
    # ------------------------------------------------------------------

    def _load_from_yaml(self) -> None:
        """Pre-populate form from the YAML files on disk."""
        from maestro.utils.config_loader import load_runtime_config

        try:
            cfg = load_runtime_config(self._config_path)
        except Exception:
            return
        self._base_app_vals = cfg.app.model_dump()
        self._base_infer_vals = cfg.inference.model_dump()
        self._populate(cfg)

    def _populate(self, cfg: RuntimeConfig) -> None:
        app_dump = cfg.app.model_dump()
        infer_dump = cfg.inference.model_dump()
        combined = {**app_dump, **infer_dump}
        # Flatten the nested gaze config so the gaze_enabled widget is reachable
        combined["gaze_enabled"] = infer_dump.get("gaze", {}).get("enabled", True)
        for name, widget in self._widgets.items():
            val = combined.get(name)
            if val is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(val))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)

    def _on_apply(self) -> None:
        """Write validated config to YAML, then emit apply_requested."""
        cfg = self.current_config()
        if cfg is None or self._runtime_active:
            return
        app_path = self._config_path / "app.yaml"
        infer_path = self._config_path / "inference.yaml"
        _safe_write(app_path, cfg.app.model_dump())
        _safe_write(infer_path, cfg.inference.model_dump())
        self.apply_requested.emit()


def _safe_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    tmp.replace(path)


def _ge(info: Any) -> int | float | None:
    for m in info.metadata:
        if hasattr(m, "ge"):
            return m.ge
    return None


def _le(info: Any) -> int | float | None:
    for m in info.metadata:
        if hasattr(m, "le"):
            return m.le
    return None
