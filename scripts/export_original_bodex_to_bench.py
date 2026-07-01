from __future__ import annotations

"""Copy original BODex grasps into DexGraspBench output layout with host paths."""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="../external/DexGraspBench/output/bodex_sparse_official_shadow/graspdata")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", nargs="+", default=["battery", "mouse", "rubiks", "sodacan"])
    return parser.parse_args()


def host_path_text(path_text: str) -> str:
    text = str(path_text)
    if text.startswith("/mnt/e/"):
        return str(Path("E:/" + text[len("/mnt/e/") :]).resolve())
    return text


def main() -> None:
    args = parse_args()
    total = 0
    for object_name in args.objects:
        source = Path(args.source_root) / object_name / "full_static" / "0.npy"
        data = dict(np.load(source, allow_pickle=True).item())
        data["obj_path"] = host_path_text(data["obj_path"])
        out_path = Path(args.output_root) / object_name / "5f_full" / "0.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data)
        total += 1
    print(f"Exported {total} original BODex grasps to {args.output_root}")


if __name__ == "__main__":
    main()
