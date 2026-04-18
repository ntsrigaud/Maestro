from __future__ import annotations

import argparse

from gesture_control.app.bootstrap import build_container
from gesture_control.app.runtime import Runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Presentation and meeting control runtime scaffold"
    )
    parser.add_argument(
        "--config",
        default="configs",
        help="Path to configuration directory that contains app.yaml and inference.yaml",
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
        "--registry",
        default=None,
        help=(
            "[Deprecated] Local model registry path for backend=local. "
            "Configure model_source in configs/model_source.yaml instead."
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
    container = build_container(
        args.config,
        model_artifact_path=args.model_artifact,
        registry_path=args.registry,
        hf_revision_override=args.hf_revision,
        show_window=not args.no_display,
    )
    runtime = Runtime(container)
    runtime.run(benchmark_output=args.benchmark_output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
