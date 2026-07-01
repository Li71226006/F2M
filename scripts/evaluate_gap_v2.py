from __future__ import annotations

"""Evaluate saved F2M runs with the responsibility-gap v2 metric.

The optimizer still writes the legacy disabled-residual metric because it is
useful for continuity with earlier experiments. This script recomputes the
paper-facing metric from saved q_start/q_refined pairs:

    positive_part(T - K^T A(q))

where T keeps active fingers' original full-hand responsibility and adds the
disabled-finger responsibility that the reduced hand should compensate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.config import CFETConfig
from method.responsibility_gap import (
    active_nonthumb_rows,
    build_patch_compensation_kernel,
    compensated_coverage,
    target_responsibility,
)
from method.sequential_qp import SequentialQP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cfet_gap_v2.json")
    parser.add_argument("--converted-root", default="results/gap_v2_inputs")
    parser.add_argument("--result-roots", nargs="+", required=True)
    parser.add_argument("--objects", nargs="+", default=["battery", "mouse", "rubiks", "sodacan"])
    parser.add_argument("--modes", nargs="+", default=["4f_no_little", "3f_thumb_index_middle", "2f_thumb_index"])
    parser.add_argument("--output", default="results/gap_v2_metric_compare.csv")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_bridge_meta(converted_dir: Path) -> dict[str, Any]:
    meta_path = converted_dir / "bridge_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing BODex bridge metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def make_runner(config_path: str, object_name: str, converted_dir: Path, device: str) -> SequentialQP:
    meta = load_bridge_meta(converted_dir)
    config = CFETConfig.from_json(
        config_path,
        object_name=f"bodex_{object_name}_eval",
        urdf_path=meta["urdf_path"],
        robot_link_pc_path=meta.get("robot_link_pc_path", ""),
        object_point_cloud=str(converted_dir / "object_pc_normals.pt"),
        initial_q_path=str(converted_dir / "q_from_bodex.pt"),
        modes=[],
        output_dir="results/_eval_unused",
        device=device,
        render=False,
    )
    return SequentialQP(config)


def coverage_metrics(runner: SequentialQP, q: torch.Tensor, finger_mask: list[int], kernel: np.ndarray) -> dict[str, float]:
    full_c, _, _, _ = runner.responsibility.compute(runner.q_full)
    current_c, _, _, _ = runner.responsibility.compute(q)
    disabled = runner.responsibility.disabled_responsibility(runner.q_full, finger_mask)
    rows = active_nonthumb_rows(finger_mask)
    active_current = current_c[rows].sum(axis=0) if rows else np.zeros_like(disabled)
    active_full = full_c[rows].sum(axis=0) if rows else np.zeros_like(disabled)

    target = target_responsibility(
        full_c,
        disabled,
        finger_mask,
        disabled_weight=runner.args.disabled_compensation_weight,
    )
    compensated = compensated_coverage(active_current, kernel)
    legacy_gap = np.maximum(0.0, disabled - active_current)
    disabled_kernel_gap = np.maximum(0.0, disabled - compensated)
    target_kernel_gap = np.maximum(0.0, target - compensated)

    active_full_mass = float(active_full.sum())
    if active_full_mass <= 1e-12:
        self_retention = 1.0
    else:
        self_retention = float(np.minimum(compensated, active_full).sum() / (active_full_mass + 1e-12))

    target_mass = float(target.sum())
    disabled_mass = float(disabled.sum())
    return {
        "legacy_gap": float(legacy_gap.sum()),
        "legacy_coverage": float(1.0 - legacy_gap.sum() / (disabled_mass + 1e-12)) if disabled_mass > 0 else 1.0,
        "disabled_kernel_gap": float(disabled_kernel_gap.sum()),
        "disabled_kernel_coverage": float(1.0 - disabled_kernel_gap.sum() / (disabled_mass + 1e-12))
        if disabled_mass > 0
        else 1.0,
        "target_kernel_gap": float(target_kernel_gap.sum()),
        "target_kernel_coverage": float(1.0 - target_kernel_gap.sum() / (target_mass + 1e-12))
        if target_mass > 0
        else 1.0,
        "self_retention": self_retention,
        "disabled_mass": disabled_mass,
        "target_mass": target_mass,
    }


def evaluate_mode(
    runner: SequentialQP,
    result_root: Path,
    object_name: str,
    mode: str,
    label: str,
    kernel: np.ndarray,
) -> dict[str, Any] | None:
    q_path = result_root / object_name / mode / "q_result.pt"
    if not q_path.exists():
        return None
    q_data = torch.load(q_path, map_location=runner.robot.device)
    q_start = q_data["q_start"].to(runner.robot.device)
    q_refined = q_data["q_refined"].to(runner.robot.device)
    finger_mask = runner.robot.finger_mask(mode)
    before = coverage_metrics(runner, q_start, finger_mask, kernel)
    after = coverage_metrics(runner, q_refined, finger_mask, kernel)
    return {
        "method": label,
        "object": object_name,
        "mode": mode,
        "legacy_gain": before["legacy_gap"] - after["legacy_gap"],
        "disabled_kernel_gain": before["disabled_kernel_gap"] - after["disabled_kernel_gap"],
        "target_kernel_gain": before["target_kernel_gap"] - after["target_kernel_gap"],
        "self_retention_after": after["self_retention"],
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
    }


def main() -> None:
    args = parse_args()
    converted_root = Path(args.converted_root)
    rows: list[dict[str, Any]] = []

    for object_name in args.objects:
        runner = make_runner(args.config, object_name, converted_root / object_name / "_converted", args.device)
        kernel = build_patch_compensation_kernel(
            runner.patch_positions,
            runner.patch_normals,
            sigma_pos=runner.args.patch_kernel_sigma_pos,
            sigma_normal=runner.args.patch_kernel_sigma_normal,
            sigma_wrench=runner.args.patch_kernel_sigma_wrench,
            wrench_weight=runner.args.patch_kernel_wrench_weight,
        )
        for result_root_arg in args.result_roots:
            result_root = Path(result_root_arg)
            label = result_root.name
            if label == "gap_v2_batch":
                label = "v2_responsibility_gap"
            elif label == "gap_v2_baseline":
                label = "baseline_point_tracking"
            for mode in args.modes:
                row = evaluate_mode(runner, result_root, object_name, mode, label, kernel)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise RuntimeError("No q_result.pt files were found for the requested roots/objects/modes.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
