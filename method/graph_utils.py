from __future__ import annotations

"""Graph utilities for link/patch responsibility.

The graph is bipartite:

    hand link surface samples <-> object surface patches

Edges are soft geometric affinities used to compare five-finger and reduced
grasps over the same object patches.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from .constants import FINGER_NAMES, SHADOWHAND_FINGER_LINK_MAP


def link_origins_by_finger(hand, q: torch.Tensor) -> tuple[dict[str, np.ndarray], list[list[str]]]:
    status = hand.pk_chain.forward_kinematics(q)
    origins = {}
    for link_name in hand.links_pc:
        if link_name in status:
            origins[link_name] = status[link_name].get_matrix()[0, :3, 3].detach().cpu().numpy()
    finger_links = []
    for finger_idx in range(len(FINGER_NAMES)):
        finger_links.append([link for link in SHADOWHAND_FINGER_LINK_MAP[finger_idx] if link in origins])
    return origins, finger_links


def build_link_patch_graph(
    hand,
    q: torch.Tensor,
    patch_positions: np.ndarray,
    *,
    top_k: int,
    sigma: float,
    radius: float,
    patch_normals: np.ndarray | None = None,
    points_per_link: int = 96,
    use_direction_factor: bool = True,
    use_signed_band: bool = False,
    outside_band: float = 0.012,
    allowed_penetration: float = 0.007,
    tangent_sigma: float | None = None,
    penetration_sigma: float = 0.012,
    saturate_contact: bool = True,
    contact_distance: float = 0.0015,
    contact_radius: float = 0.020,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build sparse link-to-patch affinity A_link.

    For every link, sampled surface points are transformed into world space.
    The legacy score used Euclidean nearest distance:

        exp(-distance^2 / sigma^2)

    With signed-band scoring enabled, a link point that reaches or penetrates
    the local neighborhood of a patch is saturated to full coverage. Otherwise
    affinity decays with distance. Deep penetration is not rewarded beyond this
    saturation; collision penalties handle it in the QP stage.

    Optionally, a direction factor relu(n_patch dot direction_to_link) keeps
    responsibility on patches that the link approaches from the outside.
    """

    status = hand.pk_chain.forward_kinematics(q)
    _, finger_links = link_origins_by_finger(hand, q)
    link_names = []
    link_fingers = []
    for finger_idx, links in enumerate(finger_links):
        for link in links:
            link_names.append(link)
            link_fingers.append(finger_idx)

    patch_t = torch.as_tensor(patch_positions, dtype=q.dtype, device=q.device)
    normal_t = None
    if patch_normals is not None:
        normal_t = torch.as_tensor(patch_normals, dtype=q.dtype, device=q.device)
        normal_t = normal_t / (torch.linalg.norm(normal_t, dim=-1, keepdim=True) + 1e-12)

    distances = np.full((len(link_names), patch_positions.shape[0]), np.inf, dtype=np.float64)
    affinities = np.zeros_like(distances, dtype=np.float64)
    tangent_sigma = sigma if tangent_sigma is None else tangent_sigma
    for link_idx, link_name in enumerate(link_names):
        if link_name not in status:
            continue
        local_pc = hand.links_pc.get(link_name)
        if local_pc is None:
            local_pc = torch.zeros((1, 3), dtype=q.dtype, device=q.device)
        else:
            local_pc = local_pc.to(q.device)
            if local_pc.shape[0] > points_per_link:
                sample_idx = torch.linspace(0, local_pc.shape[0] - 1, points_per_link, device=q.device).long()
                local_pc = local_pc[sample_idx]

        se3 = status[link_name].get_matrix()[0].to(q.device)
        ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=q.device)
        world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
        link_patch_dist = torch.cdist(world_pc[None, :, :], patch_t[None, :, :])[0]
        min_dist, closest_idx = link_patch_dist.min(dim=0)
        closest_points = world_pc[closest_idx]
        point_affinity = torch.exp(-(link_patch_dist**2) / (sigma**2))

        if use_signed_band and normal_t is not None:
            point_delta = world_pc[:, None, :] - patch_t[None, :, :]
            point_signed = (point_delta * normal_t[None, :, :]).sum(dim=-1)
            local_contact = (point_signed <= contact_distance) & (link_patch_dist <= contact_radius)
            if saturate_contact:
                point_affinity = torch.where(local_contact, torch.ones_like(point_affinity), point_affinity)
            distances[link_idx] = min_dist.detach().cpu().numpy().astype(np.float64)
        else:
            distances[link_idx] = min_dist.detach().cpu().numpy().astype(np.float64)

        if use_direction_factor and normal_t is not None:
            point_dirs = world_pc[:, None, :] - patch_t[None, :, :]
            point_dirs = point_dirs / (torch.linalg.norm(point_dirs, dim=-1, keepdim=True) + 1e-12)
            direction_factor = torch.relu((point_dirs * normal_t[None, :, :]).sum(dim=-1))
            if use_signed_band and normal_t is not None and saturate_contact:
                point_affinity = torch.where(local_contact, point_affinity, point_affinity * direction_factor)
            else:
                point_affinity = point_affinity * direction_factor

        affinities[link_idx] = point_affinity.max(dim=0).values.detach().cpu().numpy().astype(np.float64)

    weights = np.zeros_like(distances, dtype=np.float64)
    # Keep only the nearest few patches per link. This makes the graph sparse
    # and prevents one large link from assigning tiny responsibility everywhere.
    keep = min(top_k, patch_positions.shape[0])
    for link_idx in range(len(link_names)):
        nearest = np.argpartition(distances[link_idx], keep - 1)[:keep]
        if radius > 0:
            nearest = nearest[distances[link_idx, nearest] <= radius]
        if nearest.size == 0:
            continue
        weights[link_idx, nearest] = affinities[link_idx, nearest]
    return weights, link_names, link_fingers


def write_cstar_csv(path: Path, c0: np.ndarray, c_star: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["matrix", "finger", *[f"patch_{i:02d}" for i in range(c0.shape[1])], "row_sum"])
        for matrix_name, matrix in (("C0", c0), ("Cstar", c_star)):
            for finger_idx, finger in enumerate(FINGER_NAMES):
                row = matrix[finger_idx]
                writer.writerow([matrix_name, finger, *[f"{v:.6f}" for v in row], f"{row.sum():.6f}"])


def write_link_patch_csv(
    path: Path,
    link_patch: np.ndarray,
    link_names: list[str],
    link_fingers: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["link", "finger", *[f"patch_{i:02d}" for i in range(link_patch.shape[1])], "row_sum"])
        for link_idx, link_name in enumerate(link_names):
            row = link_patch[link_idx]
            writer.writerow(
                [
                    link_name,
                    FINGER_NAMES[link_fingers[link_idx]],
                    *[f"{v:.6f}" for v in row],
                    f"{row.sum():.6f}",
                ]
            )
