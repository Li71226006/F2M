from __future__ import annotations

"""Summarize DexGraspBench evaluation npy files into a CSV table."""

import argparse
import csv
from glob import glob
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        return float(item) if isinstance(item, (int, float, np.number, bool)) else item
    return arr.tolist()


def add_readable_metrics(row: dict) -> None:
    """Add human-readable aliases while keeping raw DexGraspBench keys."""

    if "succ_flag" in row:
        row["sim_success"] = row["succ_flag"]
    if "delta_pos" in row:
        row["sim_delta_pos_m"] = row["delta_pos"]
        row["sim_delta_pos_mm"] = float(row["delta_pos"]) * 1000.0
    if "delta_angle" in row:
        row["sim_delta_angle_deg"] = row["delta_angle"]
    if "ho_pene" in row:
        row["ho_pene_mm"] = float(row["ho_pene"]) * 1000.0
    if "self_pene" in row:
        row["self_pene_mm"] = float(row["self_pene"]) * 1000.0
    if "contact_dist" in row:
        row["contact_dist_mm"] = float(row["contact_dist"]) * 1000.0
    if "delta_pos" in row and "delta_angle" in row:
        row["filtered_by_initial_penetration"] = (
            float(row["delta_pos"]) >= 99.0 and float(row["delta_angle"]) >= 99.0
        )


def main() -> None:
    args = parse_args()
    rows = []
    for path_text in sorted(glob(str(Path(args.eval_root) / "**" / "*.npy"), recursive=True)):
        path = Path(path_text)
        data = np.load(path, allow_pickle=True).item()
        rel = path.relative_to(args.eval_root)
        parts = rel.parts
        row = {
            "object": parts[0] if len(parts) > 0 else "",
            "mode": parts[1] if len(parts) > 1 else "",
            "file": str(rel),
        }
        for key, value in data.items():
            if key.endswith("_qpos") or key in {"obj_pose", "obj_path"}:
                continue
            row[key] = scalar(value)
        add_readable_metrics(row)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "object",
        "mode",
        "file",
        "sim_success",
        "succ_flag",
        "filtered_by_initial_penetration",
        "sim_delta_pos_m",
        "sim_delta_pos_mm",
        "sim_delta_angle_deg",
        "ho_pene",
        "ho_pene_mm",
        "self_pene",
        "self_pene_mm",
        "contact_num",
        "contact_dist",
        "contact_dist_mm",
        "contact_consis",
        "qp_metric",
        "qp_dfc_metric",
        "dfc_metric",
        "tdg_metric",
        "q1_metric",
    ]
    keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
