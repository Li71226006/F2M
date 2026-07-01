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
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
