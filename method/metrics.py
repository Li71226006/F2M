from __future__ import annotations

import numpy as np
import torch

from .constants import FINGER_NAMES, SHADOWHAND_FINGER_LINK_MAP
from .geometry import sample_points, signed_distances

from .contact_targets import (
    active_contact_centroid,
    allowed_target_links,
    compute_finger_responsibility,
    patch_wrench_matrix,
    sampled_finger_points,
)



def reduced_penetration_stats(
    hand,
    q: torch.Tensor,
    object_pc_normals: torch.Tensor,
    finger_mask: list[int],
    args: argparse.Namespace,
) -> dict:
    active_indices = [idx for idx, enabled in enumerate(finger_mask) if enabled]
    points = sampled_finger_points(hand, q, active_indices, args.stats_points_per_link)
    if points.numel() == 0:
        return {
            "max_penetration_mm": 0.0,
            "mean_penetration_mm": 0.0,
            "penetrating_points": 0,
            "near_ratio": 0.0,
        }
    object_pc = object_pc_normals[:, :3].to(q.device)
    normals = object_pc_normals[:, 3:].to(q.device)
    signed, unsigned = signed_distances(points, object_pc, normals)
    penetration = torch.relu(-signed)
    positive = penetration[penetration > 0]
    return {
        "max_penetration_mm": float(positive.max() * 1000.0) if positive.numel() else 0.0,
        "mean_penetration_mm": float(positive.mean() * 1000.0) if positive.numel() else 0.0,
        "penetrating_points": int(positive.numel()),
        "near_ratio": float((unsigned < args.near_distance).float().mean()),
    }



def self_clearance_stats(
    hand,
    q: torch.Tensor,
    finger_mask: list[int],
    args: argparse.Namespace,
) -> dict:
    active = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
    points = {
        idx: sampled_finger_points(hand, q, [idx], args.stats_points_per_link)
        for idx in active
    }
    pair_distances = {}
    min_distance = None
    for pos, idx in enumerate(active):
        for jdx in active[pos + 1 :]:
            if points[idx].numel() == 0 or points[jdx].numel() == 0:
                continue
            dist = torch.cdist(points[idx][None, :, :], points[jdx][None, :, :])[0].min()
            value_mm = float(dist.detach().cpu() * 1000.0)
            pair_distances[f"{FINGER_NAMES[idx]}-{FINGER_NAMES[jdx]}"] = value_mm
            min_distance = value_mm if min_distance is None else min(min_distance, value_mm)
    return {
        "active_nonthumb_min_distance_mm": min_distance,
        "pair_min_distance_mm": pair_distances,
    }



def object_gap_stats(
    hand,
    q: torch.Tensor,
    object_pc_normals: torch.Tensor,
    finger_mask: list[int],
    args: argparse.Namespace,
) -> dict:
    object_pc = object_pc_normals[:, :3].to(q.device)
    normals = object_pc_normals[:, 3:].to(q.device)
    robot_pc_dict, _ = hand.get_transformed_links_pc(q)
    per_finger = {}
    for finger_idx, enabled in enumerate(finger_mask):
        if not enabled:
            continue
        chunks = [
            sample_points(robot_pc_dict[link_name], args.stats_points_per_link)
            for link_name in SHADOWHAND_FINGER_LINK_MAP[finger_idx]
            if link_name in robot_pc_dict and robot_pc_dict[link_name].numel() > 0
        ]
        if not chunks:
            continue
        points = torch.cat(chunks, dim=0)
        signed, unsigned = signed_distances(points, object_pc, normals)
        per_finger[FINGER_NAMES[finger_idx]] = {
            "min_unsigned_mm": float(unsigned.min().detach().cpu() * 1000.0),
            "min_signed_mm": float(signed.min().detach().cpu() * 1000.0),
            "near_ratio": float((unsigned < args.near_distance).float().mean().detach().cpu()),
        }
    return per_finger



def single_finger_gap_stats(
    hand,
    q: torch.Tensor,
    object_pc_normals: torch.Tensor,
    finger_idx: int,
    args: argparse.Namespace,
) -> dict:
    stats = object_gap_stats(
        hand,
        q,
        object_pc_normals,
        [1 if idx == finger_idx else 0 for idx in range(len(FINGER_NAMES))],
        args,
    )
    return stats.get(FINGER_NAMES[finger_idx], {})



def wrench_balance_stats(
    hand,
    q: torch.Tensor,
    finger_mask: list[int],
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args: argparse.Namespace,
    include_thumb: bool = True,
) -> dict:
    coverage, _, _, _ = compute_finger_responsibility(
        hand,
        q,
        patch_positions,
        patch_normals,
        args,
    )
    active_rows = [
        idx
        for idx, enabled in enumerate(finger_mask)
        if enabled and (include_thumb or idx != 0)
    ]
    if not active_rows:
        return {
            "net_force_norm": 0.0,
            "net_torque_norm": 0.0,
            "wrench_norm": 0.0,
            "coverage_mass": 0.0,
            "wrench_isotropy": 0.0,
        }
    weights = coverage[active_rows].sum(axis=0)
    mass = float(weights.sum())
    if mass <= 1e-12:
        return {
            "net_force_norm": 0.0,
            "net_torque_norm": 0.0,
            "wrench_norm": 0.0,
            "coverage_mass": 0.0,
            "wrench_isotropy": 0.0,
        }
    wrenches = patch_wrench_matrix(patch_positions, patch_normals)
    weighted = weights[:, None] * wrenches
    net = weighted.sum(axis=0) / (mass + 1e-12)
    gram = (weighted.T @ weighted) / (mass + 1e-12)
    eigvals = np.linalg.eigvalsh(gram + 1e-8 * np.eye(6))
    return {
        "net_force_norm": float(np.linalg.norm(net[:3])),
        "net_torque_norm": float(np.linalg.norm(net[3:])),
        "wrench_norm": float(np.linalg.norm(net)),
        "coverage_mass": mass,
        "wrench_isotropy": float(eigvals[0] / (eigvals[-1] + 1e-12)),
    }



def thumb_opposition_stats(
    hand,
    q: torch.Tensor,
    finger_mask: list[int],
    object_pc_normals: torch.Tensor,
    patch_positions: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    object_center = patch_positions.mean(axis=0)
    active_centroid = active_contact_centroid(hand, q, finger_mask, object_pc_normals, args)
    thumb_points = sampled_finger_points(hand, q, [0], args.stats_points_per_link)
    if active_centroid is None or thumb_points.numel() == 0:
        return {"alignment": 0.0, "span_mm": 0.0}
    thumb_centroid = thumb_points.mean(dim=0).detach().cpu().numpy().astype(np.float64)
    active_vec = active_centroid - object_center
    thumb_vec = thumb_centroid - object_center
    active_norm = np.linalg.norm(active_vec)
    thumb_norm = np.linalg.norm(thumb_vec)
    if active_norm < 1e-8 or thumb_norm < 1e-8:
        alignment = 0.0
    else:
        alignment = float(np.dot(active_vec / active_norm, -thumb_vec / thumb_norm))
    return {
        "alignment": alignment,
        "span_mm": float(np.linalg.norm(active_centroid - thumb_centroid) * 1000.0),
    }
