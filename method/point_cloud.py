from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry import farthest_point_indices


def load_point_cloud_with_normals(path: str | Path, device: torch.device) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    if path.suffix.lower() == ".pt":
        data = torch.load(path, map_location=device)
        if isinstance(data, dict):
            for key in ("points_normals", "object_pc_normals", "pc_normals"):
                if key in data:
                    data = data[key]
                    break
        tensor = data if torch.is_tensor(data) else torch.as_tensor(data)
    elif path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            key = "points_normals" if "points_normals" in data else data.files[0]
            data = data[key]
        tensor = torch.as_tensor(data)
    else:
        array = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
        tensor = torch.as_tensor(array)

    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] not in {3, 6}:
        raise ValueError(f"Point cloud must have shape [N,3] or [N,6], got {tuple(tensor.shape)}")
    if tensor.shape[1] == 3:
        center = tensor.mean(dim=0, keepdim=True)
        normals = tensor - center
        normals = normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-8)
        tensor = torch.cat([tensor, normals], dim=1)
    else:
        normals = tensor[:, 3:]
        tensor[:, 3:] = normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-8)
    return tensor


def make_object_patches(object_pc_normals: torch.Tensor, num_patches: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the object surface into patch centers and normals.

    These patches are the discrete object nodes O_k used by responsibility
    assignment, target selection, and wrench proxy scoring.
    """

    object_pc = object_pc_normals[:, :3]
    normals = object_pc_normals[:, 3:]
    idx = farthest_point_indices(object_pc, num_patches)
    return (
        object_pc[idx].detach().cpu().numpy().astype(np.float64),
        normals[idx].detach().cpu().numpy().astype(np.float64),
    )
