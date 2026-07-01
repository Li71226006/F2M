from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

F2M_ROOT = Path(__file__).resolve().parents[2]
if str(F2M_ROOT) not in sys.path:
    sys.path.insert(0, str(F2M_ROOT))

from method.func import resolve_project_path
from method.hand_model import create_hand_model
from method.pose_adapter import PoseAdapter, joint_names_from_mjcf


DEFAULT_BODEX_SHADOW_XML = "external/DexGraspBench/assets/hand/shadow/right_hand.xml"


def _host_path(path_text: str) -> Path:
    """把 BODex 输出里的 WSL 路径映射回当前 Windows workspace。"""

    text = str(path_text)
    if text.startswith("/mnt/e/"):
        return Path("E:/" + text[len("/mnt/e/") :])
    return resolve_project_path(text)


def _load_bodex_dict(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if data.shape != ():
        raise ValueError(f"Expected a scalar npy dict from BODex, got shape {data.shape}")
    return data.item()


def _sample_object_pc(grasp_data: dict, num_points: int) -> torch.Tensor:
    """从 BODex 的 object mesh 采样 F2M 使用的 [xyz, normal] 点云。"""

    obj_root = _host_path(grasp_data["obj_path"])
    mesh_path = obj_root if obj_root.suffix.lower() == ".obj" else obj_root / "mesh" / "simplified.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"BODex object mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, force="mesh")
    scale = float(grasp_data.get("obj_scale", 1.0))
    mesh.apply_scale(scale)
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_idx]
    points_normals = np.concatenate([points, normals], axis=1).astype(np.float32)
    return torch.as_tensor(points_normals)


def _bodex_hand_vertices(source_xml: str | Path, qpos: np.ndarray) -> np.ndarray:
    """用 BODex 的 MJCF 手模型计算同一个 qpos 下的手部顶点。

    这一步用于标定 BODex MJCF hand root 和 F2M URDF hand root 的平移差。
    BODex 的 qpos 是 root pose + 22 个手关节；MJCF 文件本身只吃 22 个关节。
    """

    import mujoco

    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    root_xyz = qpos[:3]
    root_quat_wxyz = qpos[3:7]
    hand_qpos = qpos[7:]
    root_rot = Rotation.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]).as_matrix()

    model = mujoco.MjModel.from_xml_path(str(resolve_project_path(source_xml)))
    data = mujoco.MjData(model)
    if data.qpos.shape[0] != hand_qpos.shape[0]:
        raise ValueError(f"BODex MJCF expects {data.qpos.shape[0]} qpos values, got {hand_qpos.shape[0]}")
    data.qpos[:] = hand_qpos
    mujoco.mj_kinematics(model, data)

    chunks = []
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vert_start : vert_start + vert_count]
        geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
        geom_trans = data.geom_xpos[geom_id]
        posed = (vertices @ geom_rot.T + geom_trans) @ root_rot.T + root_xyz
        chunks.append(posed)
    if not chunks:
        raise ValueError(f"No mesh vertices found in BODex MJCF: {source_xml}")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def convert_bodex_output(
    bodex_npy: str | Path,
    urdf_path: str | Path,
    output_dir: str | Path,
    *,
    robot_link_pc_path: str | Path | None = None,
    source_xml: str | Path = DEFAULT_BODEX_SHADOW_XML,
    qpos_key: str = "grasp_qpos",
    object_points: int = 2048,
    root_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_centers: bool = False,
    calibrate_hand_frame: bool = False,
) -> dict:
    """把一个 BODex .npy 输出转换成 F2M 的 q.pt 和 object_pc_normals.pt。"""

    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grasp_data = _load_bodex_dict(bodex_npy)
    source_joint_names = joint_names_from_mjcf(resolve_project_path(source_xml))

    hand = create_hand_model(
        "shadowhand",
        resolve_project_path(urdf_path),
        torch.device("cpu"),
        resolve_project_path(robot_link_pc_path) if robot_link_pc_path else None,
    )
    adapter = PoseAdapter(hand.pk_chain.get_joint_parameter_names()[6:])
    q = adapter.from_bodex_qpos(grasp_data[qpos_key], source_joint_names)
    q[:3] += torch.as_tensor(root_xyz_offset, dtype=q.dtype)
    object_pc_normals = _sample_object_pc(grasp_data, object_points)

    if calibrate_hand_frame:
        # 更合理的自动对齐：用 BODex 自己的 MJCF/FK 得到手 mesh，
        # 再把 F2M URDF 手模型的中心平移到 BODex 手 mesh 中心。
        robot_pc_dict, _ = hand.get_transformed_links_pc(q)
        f2m_hand_points = torch.cat(list(robot_pc_dict.values()), dim=0)
        bodex_hand_points = torch.as_tensor(
            _bodex_hand_vertices(resolve_project_path(source_xml), grasp_data[qpos_key]),
            dtype=q.dtype,
        )
        q[:3] += bodex_hand_points.mean(dim=0) - f2m_hand_points.mean(dim=0)

    if align_centers:
        # 调试坐标系对齐时使用：只改根节点平移，不改姿态和关节角。
        robot_pc_dict, _ = hand.get_transformed_links_pc(q)
        hand_points = torch.cat(list(robot_pc_dict.values()), dim=0)
        q[:3] += object_pc_normals[:, :3].mean(dim=0) - hand_points.mean(dim=0)

    q_path = output_dir / "q_from_bodex.pt"
    pc_path = output_dir / "object_pc_normals.pt"
    meta_path = output_dir / "bridge_meta.json"
    torch.save({"q": q, "source": str(resolve_project_path(bodex_npy)), "qpos_key": qpos_key}, q_path)
    torch.save(object_pc_normals, pc_path)

    meta = {
        "source_bodex_npy": str(resolve_project_path(bodex_npy)),
        "source_xml": str(resolve_project_path(source_xml)),
        "urdf_path": str(resolve_project_path(urdf_path)),
        "qpos_key": qpos_key,
        "bodex_qpos_dim": int(np.asarray(grasp_data[qpos_key]).size),
        "f2m_q_dim": int(q.numel()),
        "root_xyz_offset": list(root_xyz_offset),
        "align_centers": bool(align_centers),
        "calibrate_hand_frame": bool(calibrate_hand_frame),
        "source_joint_names": source_joint_names,
        "internal_joint_names": hand.pk_chain.get_joint_parameter_names(),
        "q_path": str(q_path),
        "object_point_cloud": str(pc_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a BODex .npy grasp into F2M inputs.")
    parser.add_argument("--bodex-npy", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--robot-link-pc")
    parser.add_argument("--source-xml", default=DEFAULT_BODEX_SHADOW_XML)
    parser.add_argument("--qpos-key", default="grasp_qpos", choices=["pregrasp_qpos", "grasp_qpos", "squeeze_qpos"])
    parser.add_argument("--object-points", type=int, default=2048)
    parser.add_argument("--root-xyz-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--align-centers", action="store_true")
    parser.add_argument("--calibrate-hand-frame", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    meta = convert_bodex_output(
        args.bodex_npy,
        args.urdf,
        args.output_dir,
        robot_link_pc_path=args.robot_link_pc,
        source_xml=args.source_xml,
        qpos_key=args.qpos_key,
        object_points=args.object_points,
        root_xyz_offset=tuple(args.root_xyz_offset),
        align_centers=args.align_centers,
        calibrate_hand_frame=args.calibrate_hand_frame,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
