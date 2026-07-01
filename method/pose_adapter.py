from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _clean_joint_name(name: str) -> str:
    """把 MuJoCo 常见的 rh_/lh_ 前缀去掉，方便和 URDF 关节名对齐。"""

    for prefix in ("rh_", "lh_", "hand:rh_", "hand:lh_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def joint_names_from_mjcf(xml_path: str | Path) -> list[str]:
    """从 MJCF/XML 中按文件顺序读出 hinge/slide/free 关节名。"""

    root = ET.parse(xml_path).getroot()
    names: list[str] = []
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if name:
            names.append(_clean_joint_name(name))
    return names


def quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """BODex/DexGraspBench 的四元数是 wxyz，scipy 需要 xyzw。"""

    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("XYZ")


@dataclass(frozen=True)
class PoseAdapter:
    """把外部方法的姿态统一转成 F2M 内部 q。

    F2M 内部约定：
    - 前 6 维是虚拟根节点：x, y, z, roll, pitch, yaw
    - 后面按 URDF/FK 的 joint_names 顺序排列
    """

    internal_joint_names: list[str]

    def from_root_quat_and_named_joints(
        self,
        root_xyz: np.ndarray,
        root_quat_wxyz: np.ndarray,
        source_joint_values: np.ndarray,
        source_joint_names: list[str],
        *,
        missing_joint_value: float = 0.0,
    ) -> torch.Tensor:
        """用于 BODex 这种 xyz + quat + named joints 的格式。"""

        root_xyz = np.asarray(root_xyz, dtype=np.float64)
        root_rpy = quat_wxyz_to_euler_xyz(root_quat_wxyz)
        source = {
            _clean_joint_name(name): float(value)
            for name, value in zip(source_joint_names, np.asarray(source_joint_values, dtype=np.float64))
        }

        values = np.full(len(self.internal_joint_names), missing_joint_value, dtype=np.float64)
        for idx, name in enumerate(self.internal_joint_names):
            values[idx] = source.get(_clean_joint_name(name), missing_joint_value)

        q = np.concatenate([root_xyz, root_rpy, values], axis=0)
        return torch.as_tensor(q, dtype=torch.float32)

    def from_bodex_qpos(
        self,
        qpos: np.ndarray,
        source_joint_names: list[str],
        *,
        missing_joint_value: float = 0.0,
    ) -> torch.Tensor:
        """BODex qpos: xyz(3) + quat_wxyz(4) + hand joints。"""

        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size < 8:
            raise ValueError(f"BODex qpos is too short: {qpos.size}")
        return self.from_root_quat_and_named_joints(
            qpos[:3],
            qpos[3:7],
            qpos[7:],
            source_joint_names,
            missing_joint_value=missing_joint_value,
        )
