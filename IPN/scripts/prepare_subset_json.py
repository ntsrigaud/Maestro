"""
Converts IPN annotation CSVs into the JSON format expected by the IPN-hand training repo,
filtered to a custom subset of gesture classes.

Usage (run from the IPN-hand repo root on the server):
    python scripts/prepare_subset_json.py \
        --annot_dir ~/.Neil/Gesture-Control-System/data/IPN/annotations \
        --output_json annotation_ipnGesture/ipn_subset.json

Outputs:
    annotation_ipnGesture/ipn_subset.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict

# ── Your gesture subset mapped to IPN class codes ─────────────────────────────
# Unknown        → D0X  (Non-gesture)
# Point_one      → B0A  (Pointing with one finger)
# Point_two      → B0B  (Pointing with two fingers)
# Swipe_up       → G03  (Throw up)
# Swipe_down     → G04  (Throw down)
# Swipe_left     → G05  (Throw left)
# Swipe_right    → G06  (Throw right)
# Zoom_in        → G10
# Zoom_out       → G11
# NOTE: Open_palm_hold has no direct IPN equivalent and is excluded.
#       Add "G07" here (Open twice) if you want to include a stand-in.
SUBSET_CLASSES = ["D0X", "B0A", "B0B", "G03", "G04", "G05", "G06", "G10", "G11"]


def read_annot(filepath, subset_name, allowed_classes):
    entries = []
    with open(filepath, newline="") as f:
        # Peek at first line to detect whether a header row is present
        first_line = f.readline()
        f.seek(0)
        has_header = first_line.startswith("video,")
        fieldnames = ["video", "label", "id", "t_start", "t_end", "frames"]
        reader = csv.DictReader(f, fieldnames=None if has_header else fieldnames)
        for row in reader:
            label = row["label"].strip()
            if label in allowed_classes:
                entries.append({
                    "video":   row["video"].strip(),
                    "label":   label,
                    "t_start": row["t_start"].strip(),
                    "t_end":   row["t_end"].strip(),
                    "subset":  subset_name,
                })
    return entries


def build_database(entries):
    """
    Multiple gesture instances from the same video get keys like:
      ./frames/VIDEO_NAME        (first instance)
      ./frames/VIDEO_NAME^1      (second instance)
      ./frames/VIDEO_NAME^2      (third instance)  ...
    This matches the convention used in the original IPN JSON annotations.
    """
    database = {}
    instance_count = defaultdict(int)

    for e in entries:
        # Keys must be bare video names (no "./frames/" prefix) because the
        # IPN dataset class does: os.path.join(root_path, key.split('^')[0])
        # and root_path is already the frames directory.
        base_key = e["video"]
        idx = instance_count[base_key]
        key = base_key if idx == 0 else f"{base_key}^{idx}"
        instance_count[base_key] += 1

        database[key] = {
            "subset": e["subset"],
            "annotations": {
                "label":       e["label"],
                "start_frame": e["t_start"],
                "end_frame":   e["t_end"],
            },
        }
    return database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot_dir",
        required=True,
        help="Path to the IPN annotations folder (contains Annot_TrainList.txt etc.)",
    )
    parser.add_argument(
        "--output_json",
        default="annotation_ipnGesture/ipn_subset.json",
        help="Where to write the output JSON (default: annotation_ipnGesture/ipn_subset.json)",
    )
    args = parser.parse_args()

    train_file = os.path.join(args.annot_dir, "Annot_TrainList.txt")
    test_file  = os.path.join(args.annot_dir, "Annot_TestList.txt")

    for f in [train_file, test_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Annotation file not found: {f}")

    print(f"Reading training annotations from: {train_file}")
    train_entries = read_annot(train_file, "training",   SUBSET_CLASSES)

    print(f"Reading test annotations from:     {test_file}")
    test_entries  = read_annot(test_file,  "validation", SUBSET_CLASSES)

    all_entries = train_entries + test_entries
    database    = build_database(all_entries)

    output = {"labels": SUBSET_CLASSES, "database": database}

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f)

    n_train = sum(1 for v in database.values() if v["subset"] == "training")
    n_val   = sum(1 for v in database.values() if v["subset"] == "validation")

    print()
    print(f"Output JSON:  {args.output_json}")
    print(f"Labels ({len(SUBSET_CLASSES)}): {SUBSET_CLASSES}")
    print(f"Total clips:  {len(database)}")
    print(f"  Training:   {n_train}")
    print(f"  Validation: {n_val}")


if __name__ == "__main__":
    main()
