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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container = build_container(args.config)
    runtime = Runtime(container)
    runtime.run()


if __name__ == "__main__":
    main()
