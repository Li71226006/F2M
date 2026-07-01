from __future__ import annotations

"""Run DexGraspBench evaluation serially for exported F2M graspdata."""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

from hydra import compose, initialize_config_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexgraspbench-root", default="../external/DexGraspBench")
    parser.add_argument("--exp-name", default="f2m_v2b")
    parser.add_argument("--max-num", type=int, default=-1)
    parser.add_argument("--simulation", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = Path(args.dexgraspbench_root).resolve()
    src_root = bench_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    os.chdir(bench_root)

    with initialize_config_dir(config_dir=str(bench_root / "config"), version_base=None):
        overrides = [
            "hand=shadow",
            "task=eval",
            f"exp_name={args.exp_name}",
            f"task.max_num={args.max_num}",
            "n_worker=1",
            "skip=false",
        ]
        if not args.simulation:
            overrides.append("task.simulation_metrics=null")
        cfg = compose(config_name="base", overrides=overrides)

    cfg.hand_name = "shadow"
    cfg.task_name = "eval"
    cfg.save_dir = str(bench_root / "output" / f"{args.exp_name}_shadow")
    cfg.grasp_dir = str(Path(cfg.save_dir) / "graspdata")
    cfg.eval_dir = str(Path(cfg.save_dir) / "evaluation")
    cfg.succ_dir = str(Path(cfg.save_dir) / "succgrasp")
    cfg.task.debug_dir = str(Path(cfg.save_dir) / "debug")
    cfg.hand.xml_path = str(bench_root / cfg.hand.xml_path)

    from task.eval_func.fc_mocap import fcMocapEval

    grasp_paths = sorted(glob(str(Path(cfg.grasp_dir) / "**" / "*.npy"), recursive=True))
    if cfg.task.max_num > 0:
        grasp_paths = grasp_paths[: cfg.task.max_num]
    print(f"Serial DexGraspBench eval: {len(grasp_paths)} grasps from {cfg.grasp_dir}")

    ok = 0
    failed = 0
    for grasp_path in grasp_paths:
        try:
            fcMocapEval(grasp_path, cfg).run()
            ok += 1
            print(f"[ok] {grasp_path}")
        except Exception as exc:
            failed += 1
            print(f"[failed] {grasp_path}: {type(exc).__name__}: {exc}")
    print(f"Done. evaluated={ok}, failed={failed}, eval_dir={cfg.eval_dir}")


if __name__ == "__main__":
    main()
