"""
Reads per-epoch validation logs produced by main.py and prints a comparison
table of best top-1 accuracy, precision, recall, and parameter count.

Usage (run from the IPN-hand repo root on the server after training):
    python scripts/collect_results.py --result_dir results_subset
"""

import argparse
import os
import csv
import torch
from torch import nn

# ── Model metadata: (store_name_prefix, model, depth, shortcut, modality,
#                     sample_dur, display_name)
MODELS = [
    ("resnext101_RGB",    "resnext",    101, "B", "RGB",    32, "ResNeXt-101  RGB"),
    ("resnet50_RGB",      "resnet",      50, "B", "RGB",    16, "ResNet-50    RGB"),
    ("resnet10_RGB",      "resnet",      10, "B", "RGB",    16, "ResNet-10    RGB"),
    ("resnetl10_RGB",     "resnetl",     10, "A", "RGB",    16, "ResNetL-10   RGB"),
    ("c3d_RGB",           "c3d",         10, "B", "RGB",    16, "C3D          RGB"),
    ("mobilenetv2_RGB",   "mobilenetv2",  0, "B", "RGB",    16, "MobileNetV2  RGB"),
    ("shufflenetv2_RGB",  "shufflenetv2", 0, "B", "RGB",    16, "ShuffleNetV2 RGB"),
]

N_CLASSES   = 9
SAMPLE_SIZE = 112


def count_params(model_name, model_depth, shortcut, modality, sample_dur):
    """Instantiate the model and count trainable parameters."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        if model_name == "resnet":
            from models import resnet
            if model_depth == 10:
                m = resnet.resnet10(N_CLASSES, shortcut, SAMPLE_SIZE, sample_dur)
            else:
                m = resnet.resnet50(N_CLASSES, shortcut, SAMPLE_SIZE, sample_dur)
        elif model_name == "resnetl":
            from models import resnetl
            m = resnetl.resnetl10(N_CLASSES, shortcut, SAMPLE_SIZE, sample_dur)
        elif model_name == "resnext":
            from models import resnext
            m = resnext.resnet101(N_CLASSES, shortcut, cardinality=32,
                                  sample_size=SAMPLE_SIZE, sample_duration=sample_dur)
        elif model_name == "c3d":
            from models import c3d
            m = c3d.c3d_v1(SAMPLE_SIZE, sample_dur, N_CLASSES)
        elif model_name == "mobilenetv2":
            from models import mobilenetv2
            m = mobilenetv2.mob_v2(N_CLASSES, SAMPLE_SIZE, width_mult=1.0)
        elif model_name == "shufflenetv2":
            from models import shufflenetv2
            m = shufflenetv2.shf_v2(N_CLASSES, SAMPLE_SIZE, width_mult=1.0)
        else:
            return None
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total
    except Exception as e:
        return None


def read_val_log(log_path):
    """Return (best_acc, best_precision, best_recall, best_epoch) from a val log."""
    best_acc  = 0.0
    best_prec = 0.0
    best_rec  = 0.0
    best_ep   = 0
    try:
        with open(log_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    acc = float(row["acc"])
                    if acc > best_acc:
                        best_acc  = acc
                        best_prec = float(row.get("precision", 0))
                        best_rec  = float(row.get("recall", 0))
                        best_ep   = int(row["epoch"])
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return best_acc, best_prec, best_rec, best_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results_subset",
                        help="Directory containing val_*.log files")
    args = parser.parse_args()

    rows = []
    for (store, model, depth, shortcut, modality, dur, display) in MODELS:
        # val log is named: val_{store_name}_{arch}.log  (store_name already has arch in it
        # via opt.store_name = '{}_{}'.format(store_name, arch) in main.py)
        arch      = f"{model}-{depth}" if depth > 0 else model
        full_name = f"{store}_{arch}"
        log_path  = os.path.join(args.result_dir, f"val_{full_name}.log")

        acc, prec, rec, ep = read_val_log(log_path)
        params = count_params(model, depth, shortcut, modality, dur)
        params_str = f"~{params/1e6:.1f}M" if params else "N/A"

        rows.append((display, acc, prec, rec, ep, params_str, log_path))

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = [22, 10, 11, 9, 8, 10, 6]
    header = ["Model", "Top-1 Acc", "Precision", "Recall", "Params", "Best Ep", "Log"]
    sep    = "  ".join("-" * w for w in col_w)

    print()
    print("  Gesture Subset (9 classes) — Comparison Table")
    print("=" * 80)
    print("  {:<22}  {:>10}  {:>11}  {:>9}  {:>10}  {:>8}".format(*header[:6]))
    print("  " + sep)
    for (display, acc, prec, rec, ep, params_str, log_path) in rows:
        status = "NOT TRAINED" if acc == 0.0 and ep == 0 else ""
        print("  {:<22}  {:>9.2f}%  {:>10.2f}%  {:>8.2f}%  {:>10}  {:>8}  {}".format(
            display,
            acc  * 100,
            prec * 100,
            rec  * 100,
            params_str,
            ep,
            status,
        ))
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
