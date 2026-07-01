from __future__ import annotations

from dataclasses import dataclass

from .config import CFETConfig
from .func import build_solver_args, resolve_project_path
from .point_cloud import load_point_cloud_with_normals, make_object_patches


@dataclass
class ObjectProcessor:
    """Object point-cloud and patch processing."""

    config: CFETConfig

    def __post_init__(self) -> None:
        if not self.config.object_point_cloud:
            raise ValueError("F2M needs config.object_point_cloud or --point-cloud for the local method.")
        self.args = build_solver_args(self.config)

    def load_point_cloud_with_normals(self, device):
        return load_point_cloud_with_normals(resolve_project_path(self.config.object_point_cloud), device)

    def make_patches(self, object_pc_normals):
        return make_object_patches(object_pc_normals, self.config.num_patches)
