from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gesture_control.utils.schema_validation import ModelSourceConfig


class ModelPanel(QWidget):
    """
    Two-tab model source selector.

    Tab 0 — Hugging Face: repo_id, revision, filename fields.
    Tab 1 — Local: file picker for .pt artifacts.

    Call artifact_path() / hf_revision_override() to read the selection
    before passing to RuntimeController.start().
    """

    def __init__(
        self,
        default_source: ModelSourceConfig | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        src = default_source or ModelSourceConfig()

        tabs = QTabWidget()

        # --- Tab 0: Hugging Face ---
        hf_tab = QWidget()
        hf_form = QFormLayout(hf_tab)
        self._repo_id = QLineEdit(src.repo_id)
        self._revision = QLineEdit(src.revision)
        self._filename = QLineEdit(src.filename)
        hf_form.addRow("Repo ID:", self._repo_id)
        hf_form.addRow("Revision:", self._revision)
        hf_form.addRow("Filename:", self._filename)
        tabs.addTab(hf_tab, "Hugging Face")

        # --- Tab 1: Local ---
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        self._local_path_label = QLabel("No file selected")
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._browse)
        local_layout.addWidget(self._local_path_label)
        local_layout.addWidget(browse_btn)
        local_layout.addStretch()
        tabs.addTab(local_tab, "Local Artifact")

        self._tabs = tabs
        self._local_path: str | None = None

        outer = QVBoxLayout(self)
        box = QGroupBox("Model Source")
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(tabs)
        outer.addWidget(box)

    # ------------------------------------------------------------------
    # Public API — read by MainWindow before calling controller.start()
    # ------------------------------------------------------------------

    def artifact_path(self) -> str | None:
        """Return the local .pt path if Local tab is active, else None."""
        if self._tabs.currentIndex() == 1:
            return self._local_path
        return None

    def hf_revision_override(self) -> str | None:
        """Return the HF revision if HF tab active and revision is non-empty."""
        if self._tabs.currentIndex() == 0:
            rev = self._revision.text().strip()
            return rev if rev else None
        return None

    def hf_repo_id(self) -> str:
        return self._repo_id.text().strip()

    def hf_filename(self) -> str:
        return self._filename.text().strip()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Artifact", "", "PyTorch Models (*.pt *.pth)"
        )
        if path:
            self._local_path = path
            self._local_path_label.setText(Path(path).name)
