from __future__ import annotations

from maestro.app.ui.model_panel import ModelPanel
from maestro.utils.schema_validation import ModelSourceConfig


class TestModelPanel:
    def test_default_hf_tab_active(self, qtbot) -> None:
        panel = ModelPanel()
        qtbot.addWidget(panel)
        assert panel._tabs.currentIndex() == 0

    def test_artifact_path_none_when_hf_tab_active(self, qtbot) -> None:
        panel = ModelPanel()
        qtbot.addWidget(panel)
        assert panel.artifact_path() is None

    def test_hf_revision_override_returns_revision(self, qtbot) -> None:
        src = ModelSourceConfig(revision="v1.2")
        panel = ModelPanel(default_source=src)
        qtbot.addWidget(panel)
        assert panel.hf_revision_override() == "v1.2"

    def test_artifact_path_returns_none_when_no_local_file_selected(self, qtbot) -> None:
        panel = ModelPanel()
        qtbot.addWidget(panel)
        panel._tabs.setCurrentIndex(1)
        assert panel.artifact_path() is None

    def test_artifact_path_returns_path_after_selection(self, qtbot) -> None:
        panel = ModelPanel()
        qtbot.addWidget(panel)
        panel._tabs.setCurrentIndex(1)
        panel._local_path = "/tmp/model.pt"
        assert panel.artifact_path() == "/tmp/model.pt"

    def test_pre_populated_from_model_source_config(self, qtbot) -> None:
        src = ModelSourceConfig(repo_id="org/custom-model", revision="exp-branch")
        panel = ModelPanel(default_source=src)
        qtbot.addWidget(panel)
        assert panel._repo_id.text() == "org/custom-model"
        assert panel._revision.text() == "exp-branch"
