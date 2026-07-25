from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from maestro.training.export_checkpoint import (
    export_inference_artifact,
    update_model_registry,
)
from maestro.training.train_lstm import load_training_config, train_lstm
from maestro.utils.config_loader import load_runtime_config
from maestro.utils.feature_validation import validate_dataset_compatibility


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.exists():
        return path

    default_path = Path("configs/model/two_stream_attn.yaml")

    # /tmp is commonly used for one-off smoke configs and gets cleaned frequently.
    if path.is_absolute() and str(path).startswith("/tmp/") and default_path.exists():
        print(
            f"Warning: config not found at {path}. Falling back to default config: {default_path}",
            file=sys.stderr,
        )
        return default_path

    raise FileNotFoundError(
        f"Config not found: {path}. "
        "Use --config configs/model/two_stream_attn.yaml or omit --config to use default."
    )


def _normalize_run_dir(run_dir: str) -> Path:
    value = run_dir.strip()
    if not value:
        raise ValueError("--run-dir must not be empty")

    # On POSIX shells, trailing ':' is often produced by zsh autocorrect.
    if value.endswith(":"):
        suggested = value.rstrip(":") or "experiments/runs"
        raise ValueError(
            "Invalid --run-dir ending with ':'. "
            f"Did zsh autocorrect your input? Try: --run-dir {suggested}"
        )

    # Allow Windows drive prefixes (e.g., C:\\runs), reject other colon usage.
    if ":" in value and not re.match(r"^[A-Za-z]:([\\\\/]|$)", value):
        raise ValueError(
            "Invalid --run-dir containing ':'. Use a normal relative or absolute directory path."
        )

    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LSTM gesture classifier and register model artifacts",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset",
        required=True,
        help="Path to sequence dataset NPZ artifact",
    )
    parser.add_argument(
        "--config",
        default="configs/model/two_stream_attn.yaml",
        help="Model/training YAML config path",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Configuration directory for runtime feature schema",
    )
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints",
        help="Directory for training checkpoints",
    )
    parser.add_argument(
        "--export-dir",
        default="models/exported",
        help="Directory for exported inference artifacts",
    )
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Path to model registry JSON",
    )
    parser.add_argument(
        "--run-dir",
        default="experiments/runs",
        help="Directory root for run logs and reports",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto | cpu | cuda | mps",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    runtime_config = load_runtime_config(args.config_dir)
    validate_dataset_compatibility(dataset_path, runtime_config.feature_pipeline)

    config_path = _resolve_config_path(args.config)
    train_cfg = load_training_config(config_path)
    device = _resolve_device(args.device)
    run_root_dir = _normalize_run_dir(args.run_dir)

    result = train_lstm(
        dataset_path=dataset_path,
        config=train_cfg,
        output_dir=args.output_dir,
        run_root_dir=run_root_dir,
        device=device,
    )

    supports_runtime_export = train_cfg.model.model_name.startswith("two_stream_attn")
    exported_path: Path | None = None
    registry_entry = None
    if supports_runtime_export:
        exported_path = export_inference_artifact(
            result=result,
            config=train_cfg,
            export_dir=args.export_dir,
        )

        training_data_version = dataset_path.stem
        # Deprecated: update_model_registry() is retained for the local development
        # workflow only.  Once backend: huggingface is validated end-to-end, remove
        # this call together with the --registry CLI argument.
        registry_entry = (
            update_model_registry(
                registry_path=args.registry,
                model_version=result.model_version,
                checkpoint_path=result.checkpoint_path,
                exported_path=exported_path,
                class_labels=result.class_labels,
                dataset_metadata=result.dataset_metadata,
                feature_schema_version=train_cfg.export.feature_schema_version,
                training_data_version=training_data_version,
            )
            if args.registry
            else None
        )

    report = {
        "status": "success",
        "model_version": result.model_version,
        "checkpoint_path": str(result.checkpoint_path),
        "exported_path": str(exported_path) if exported_path is not None else None,
        "run_dir": str(result.run_dir),
        "val_report": result.val_report.to_dict(),
        "test_report": result.test_report.to_dict(),
        "registry_entry": registry_entry,
        "runtime_export_supported": supports_runtime_export,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
