from __future__ import annotations

from dataclasses import dataclass

from gesture_control.utils.config_loader import load_runtime_config
from gesture_control.utils.schema_validation import RuntimeConfig


@dataclass(slots=True)
class AppContainer:
    config: RuntimeConfig


def build_container(config_path: str) -> AppContainer:
    config = load_runtime_config(config_path)
    return AppContainer(config=config)
