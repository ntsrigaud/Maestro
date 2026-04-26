from __future__ import annotations

import argparse
import sys

from maestro.app._paths import get_config_dir
from maestro.app.bootstrap import build_container
from maestro.app.runtime import Runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maestro presentation control runtime")
    parser.add_argument(
        "--config",
        default=None,  # resolved at runtime via get_config_dir()
        help=(
            "Path to the configuration directory containing app.yaml and inference.yaml. "
            "Defaults to the OS user config directory (~/.config/Maestro/configs on Linux, "
            "~/Library/Application Support/Maestro/configs on macOS). "
            "Override with the MAESTRO_CONFIG_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--model-artifact",
        default=None,
        help=(
            "Path to exported inference artifact (.pt). "
            "If omitted, the model source configured in configs/model_source.yaml is used."
        ),
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help=(
            "Hugging Face revision (branch, tag, or commit hash) to override "
            "the revision set in configs/model_source.yaml. "
            "Only effective when backend is 'huggingface'."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2 window display while keeping overlay rendering in pipeline.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Optional path to write latency benchmark report JSON on shutdown.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop automatically after N frames (0 means run until interrupted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(get_config_dir())
    show_window = not args.no_display

    if show_window:
        # Qt-native preview path — identical rendering component to the GUI.
        # RuntimeController runs inference in a QThread; CLIPreviewWindow
        # displays frames via PreviewWidget on the main thread.
        from PySide6.QtWidgets import QApplication

        from maestro.app.ui.cli_preview import CLIPreviewWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = CLIPreviewWindow()
        window.show()
        window.start(
            config_path,
            artifact_path=args.model_artifact,
            hf_revision_override=args.hf_revision,
        )
        sys.exit(app.exec())
    else:
        # Headless path — no Qt dependency; benchmark and max-frames available.
        container = build_container(
            config_path,
            model_artifact_path=args.model_artifact,
            hf_revision_override=args.hf_revision,
            show_window=False,
        )
        runtime = Runtime(container)
        runtime.run(
            benchmark_output=args.benchmark_output,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
