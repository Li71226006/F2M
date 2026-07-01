from __future__ import annotations

import torch


def sample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    idx = torch.linspace(
        0,
        points.shape[0] - 1,
        max_points,
        device=points.device,
    ).long()
    return points[idx]


def signed_distances(
    points: torch.Tensor,
    object_pc: torch.Tensor,
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points.unsqueeze(0), object_pc.unsqueeze(0))[0]
    min_d, idx = distances.min(dim=1)
    nearest = object_pc[idx]
    nearest_normals = normals[idx]
    sign = torch.sign(((points - nearest) * nearest_normals).sum(-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return min_d * sign, min_d


def farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    count = min(count, points.shape[0])
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    selected[0] = torch.argmax(torch.linalg.norm(points - points.mean(dim=0), dim=1))
    dist = torch.full((points.shape[0],), float("inf"), device=points.device)
    for out_idx in range(1, count):
        prev = points[selected[out_idx - 1]]
        dist = torch.minimum(dist, torch.linalg.norm(points - prev, dim=1))
        selected[out_idx] = torch.argmax(dist)
    return selected
