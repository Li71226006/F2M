from __future__ import annotations

"""Export saved F2M q_result.pt files to DexGraspBench graspdata format."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.pose_adapter import _clean_joint_name


def host_path_text(path_text: str) -> str:
    """Map BODex WSL-style paths to the current Windows workspace."""

    text = str(path_text)
    if text.startswith("/mnt/e/"):
        return str(Path("E:/" + text[len("/mnt/e/") :]).resolve())
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", default="results/gap_v2_inputs")
    parser.add_argument("--result-root", default="results/bodex_v2b_html")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", nargs="+", default=["battery", "mouse", "rubiks", "sodacan"])
    parser.add_argument("--modes", nargs="+", default=["5f_full", "4f_no_little", "3f_thumb_index_middle", "2f_thumb_index"])
    return parser.parse_args()


def load_scalar_npy_dict(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if data.shape != ():
        raise ValueError(f"Expected scalar npy dict: {path}")
    return data.item()


def f2m_q_to_bench_qpos(
    q: torch.Tensor,
    *,
    q_start: torch.Tensor,
    source_qpos: np.ndarray,
    source_joint_names: list[str],
    internal_joint_names: list[str],
) -> np.ndarray:
    """Invert the BODex->F2M pose adapter for a saved F2M pose."""

    q_np = q.detach().cpu().numpy().astype(np.float64).reshape(-1)
    q_start_np = q_start.detach().cpu().numpy().astype(np.float64).reshape(-1)
    source_qpos = np.asarray(source_qpos, dtype=np.float64).reshape(-1)

    # bodex_bridge may calibrate F2M's URDF root translation to the BODex MJCF
    # hand mesh center. Preserve the optimized F2M displacement but remove the
    # original calibration offset when exporting back to DexGraspBench qpos.
    calibration_offset = q_start_np[:3] - source_qpos[:3]
    root_xyz = q_np[:3] - calibration_offset

    quat_xyzw = Rotation.from_euler("XYZ", q_np[3:6]).as_quat()
    root_quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    internal = {
        _clean_joint_name(name): float(q_np[idx])
        for idx, name in enumerate(internal_joint_names)
    }
    source_joints = np.asarray(
        [internal.get(_clean_joint_name(name), 0.0) for name in source_joint_names],
        dtype=np.float64,
    )
    return np.concatenate([root_xyz, root_quat_wxyz, source_joints], axis=0)


def export_one_object(args: argparse.Namespace, object_name: str) -> int:
    converted_dir = Path(args.converted_root) / object_name / "_converted"
    meta = json.loads((converted_dir / "bridge_meta.json").read_text(encoding="utf-8"))
    source_data = load_scalar_npy_dict(meta["source_bodex_npy"])
    source_qpos = np.asarray(source_data[meta.get("qpos_key", "grasp_qpos")], dtype=np.float64)

    count = 0
    for mode in args.modes:
        q_path = Path(args.result_root) / object_name / mode / "q_result.pt"
        if not q_path.exists():
            continue
        q_data = torch.load(q_path, map_location="cpu")
        q_refined = q_data["q_refined"]
        q_start = q_data["q_start"]
        bench_qpos = f2m_q_to_bench_qpos(
            q_refined,
            q_start=q_start,
            source_qpos=source_qpos,
            source_joint_names=meta["source_joint_names"],
            internal_joint_names=meta["internal_joint_names"],
        )
        out_data = {
            "obj_path": host_path_text(source_data["obj_path"]),
            "obj_pose": source_data["obj_pose"],
            "obj_scale": source_data["obj_scale"],
            "pregrasp_qpos": bench_qpos.copy(),
            "grasp_qpos": bench_qpos.copy(),
            "squeeze_qpos": bench_qpos.copy(),
            "f2m_object": object_name,
            "f2m_mode": mode,
        }
        out_path = Path(args.output_root) / object_name / mode / "0.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out_data)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    total = sum(export_one_object(args, object_name) for object_name in args.objects)
    print(f"Exported {total} DexGraspBench grasp files to {args.output_root}")


if __name__ == "__main__":
    main()
