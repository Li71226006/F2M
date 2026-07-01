from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk

from .mesh_utils import load_link_geometries
from .rotation import q_rot6d_to_q_euler


class HandModel:
    def __init__(
        self,
        robot_name: str,
        urdf_path: str | Path,
        device,
        links_pc_path: str | Path | None = None,
        link_num_points: int = 512,
    ):
        self.robot_name = robot_name
        self.urdf_path = str(Path(urdf_path).resolve())
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(Path(self.urdf_path).read_text(encoding="utf-8")).to(
            dtype=torch.float32,
            device=device,
        )
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.meshes = load_link_geometries(self.urdf_path, self.pk_chain.get_link_names())
        self.links_pc = self._load_or_sample_link_points(links_pc_path, link_num_points)
        self.vertices = {
            name: torch.as_tensor(mesh.sample(link_num_points), dtype=torch.float32, device=device)
            for name, mesh in self.meshes.items()
            if len(mesh.vertices) > 0
        }
        self.frame_status = None

    def _load_or_sample_link_points(self, links_pc_path: str | Path | None, link_num_points: int):
        if links_pc_path:
            path = Path(links_pc_path)
            if path.exists():
                data = torch.load(path, map_location=self.device)
                if isinstance(data, dict):
                    data = data.get("filtered", data)
                return {name: pc.to(self.device).float() for name, pc in data.items()}
        return {
            name: torch.as_tensor(mesh.sample(link_num_points), dtype=torch.float32, device=self.device)
            for name, mesh in self.meshes.items()
            if len(mesh.vertices) > 0
        }

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(self, q=None, links_pc=None):
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        links_pc = self.links_pc if links_pc is None else links_pc

        all_pc_se3 = {}
        all_link_se3 = []
        for link_name, link_pc in links_pc.items():
            if link_name not in self.frame_status:
                continue
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            all_pc_se3[link_name] = (link_pc_homogeneous @ se3.T)[:, :3]
            all_link_se3.append(se3)

        if all_link_se3:
            return all_pc_se3, torch.stack(all_link_se3, dim=0)
        return all_pc_se3, torch.empty(0, 4, 4, dtype=q.dtype, device=q.device)

    def get_canonical_q(self):
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower, dtype=torch.float32, device=self.device) * 0.75
        canonical_q += torch.tensor(upper, dtype=torch.float32, device=self.device) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_trimesh_q(self, q):
        self.update_status(q)
        parts = {}
        vertices = []
        faces = []
        vertex_offset = 0
        for link_name, mesh in self.meshes.items():
            if link_name not in self.frame_status:
                continue
            transform = self.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            part_mesh = mesh.copy().apply_transform(transform)
            parts[link_name] = part_mesh
            if isinstance(part_mesh, trimesh.Trimesh) and len(part_mesh.vertices) > 0:
                vertices.append(part_mesh.vertices)
                faces.append(part_mesh.faces + vertex_offset)
                vertex_offset += len(part_mesh.vertices)
        visual = trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces)) if vertices else trimesh.Trimesh()
        return {"visual": visual, "parts": parts}


def create_hand_model(
    robot_name: str,
    urdf_path: str | Path,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    links_pc_path: str | Path | None = None,
    num_points: int = 512,
):
    return HandModel(robot_name, urdf_path, device, links_pc_path, num_points)
