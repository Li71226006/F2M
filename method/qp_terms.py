from __future__ import annotations

"""Reusable local-linear QP terms.

The nonlinear geometry is frozen at the current q and linearized as:

    y(q + dq) ~= y(q) + J_y dq
    phi(y + J_y dq) ~= phi(y) + grad_phi^T J_y dq

Each helper appends CVXPY objective terms and constraints to the caller's QP.
"""

import argparse

import cvxpy as cp
import numpy as np
import torch

from .constants import SHADOWHAND_FINGER_LINK_MAP
from .geometry import sample_points, signed_distances
from .kinematics import jacobian

from .contact_targets import allowed_target_links, subset_for_candidate
from .metrics import sampled_finger_points



def add_surface_collision_constraints(
    hand,
    q: torch.Tensor,
    status,
    dq,
    objective_terms: list,
    constraints: list,
    finger_idx: int,
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Limit hand-object penetration with a softened lower SDF bound.

    Constraint:
        phi(y) + n^T J_y dq + slack >= -allowed_penetration

    Here phi is a point-cloud signed-distance proxy. Positive means outside the
    object, negative means penetration. Slack keeps the QP feasible but is
    penalized heavily.
    """

    if not args.surface_collision:
        return

    device = q.device
    object_pc = object_pc_normals[:, :3].to(device)
    normals = object_pc_normals[:, 3:].to(device)
    links = [name for name in SHADOWHAND_FINGER_LINK_MAP[finger_idx] if name in status and name in hand.links_pc]
    if not links:
        return

    col_jacs = jacobian(hand.pk_chain, q, status, links)
    for link_name in links:
        local_pc = hand.links_pc[link_name].to(device)
        local_pc = sample_points(local_pc, args.surface_collision_points)
        se3 = status[link_name].get_matrix()[0].to(device)
        ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=device)
        world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
        signed, _ = signed_distances(world_pc, object_pc, normals)
        keep = min(args.surface_collision_k, signed.numel())
        candidate_idx = torch.topk(args.surface_collision_margin - signed, keep, largest=True).indices

        frame_xyz = se3[:3, 3].detach().cpu().numpy().astype(np.float64)
        link_jac = col_jacs[link_name][0].detach().cpu().numpy().astype(np.float64)
        jv = link_jac[:3]
        jw = link_jac[3:]

        for point_idx in candidate_idx.detach().cpu().tolist():
            phi = float(signed[point_idx].detach().cpu())
            if phi > args.surface_collision_margin:
                continue
            distances = torch.cdist(world_pc[point_idx][None, None, :], object_pc[None, :, :])[0, 0]
            nearest_idx = int(torch.argmin(distances).detach().cpu())
            normal = normals[nearest_idx].detach().cpu().numpy().astype(np.float64)
            point_xyz = world_pc[point_idx].detach().cpu().numpy().astype(np.float64)
            offset = point_xyz - frame_xyz
            skew_offset = np.array(
                [
                    [0.0, -offset[2], offset[1]],
                    [offset[2], 0.0, -offset[0]],
                    [-offset[1], offset[0], 0.0],
                ],
                dtype=np.float64,
            )
            point_jac = jv - skew_offset @ jw
            slack = cp.Variable(nonneg=True)
            constraints.append(
                phi + normal @ point_jac @ dq + slack >= -args.allowed_penetration
            )
            objective_terms.append(args.collision_slack_weight * cp.sum_squares(slack))



def point_jacobian_from_link(
    se3: torch.Tensor,
    link_jac: np.ndarray,
    point_xyz: np.ndarray,
) -> np.ndarray:
    """Convert a frame spatial Jacobian into a world-point Jacobian.

    If a link frame has linear/angular Jacobian [Jv, Jw], a point offset r from
    that frame moves as:

        J_point = Jv - skew(r) Jw
    """

    frame_xyz = se3[:3, 3].detach().cpu().numpy().astype(np.float64)
    jv = link_jac[:3]
    jw = link_jac[3:]
    offset = point_xyz - frame_xyz
    skew_offset = np.array(
        [
            [0.0, -offset[2], offset[1]],
            [offset[2], 0.0, -offset[0]],
            [-offset[1], offset[0], 0.0],
        ],
        dtype=np.float64,
    )
    return jv - skew_offset @ jw



def add_contact_band_constraints(
    hand,
    q: torch.Tensor,
    status,
    dq,
    objective_terms: list,
    constraints: list,
    finger_idx: int,
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Keep the moving finger near the object surface without forcing contact.

    Despite the legacy name "contact band", this is a near-surface band:

        phi(y) <= contact_band_max_distance

    and, if enabled, the penetration-side lower bound:

        phi(y) >= -allowed_penetration

    Together they make -d_allow <= phi(y) <= d_band. This is a geometric guard,
    not a physical contact-force constraint.
    """

    if not args.contact_band:
        return

    device = q.device
    object_pc = object_pc_normals[:, :3].to(device)
    normals = object_pc_normals[:, 3:].to(device)
    links = [
        name
        for name in allowed_target_links(finger_idx, args.contact_band_link_scope)
        if name in status and name in hand.links_pc
    ]
    if not links:
        return

    band_jacs = jacobian(hand.pk_chain, q, status, links)
    for link_name in links:
        local_pc = sample_points(hand.links_pc[link_name].to(device), args.contact_band_points)
        if local_pc.numel() == 0:
            continue
        se3 = status[link_name].get_matrix()[0].to(device)
        ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=device)
        world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
        signed, unsigned = signed_distances(world_pc, object_pc, normals)
        keep = min(args.contact_band_k, unsigned.numel())
        if args.contact_band_candidate_mode == "nearest":
            candidate_idx = torch.topk(-unsigned, keep, largest=True).indices
        else:
            # Contact-band targets should be points that can plausibly approach the
            # exterior surface. If we pick deeply penetrating points, the "stay near"
            # upper bound can be satisfied by bad geometry and fight the collision term.
            exterior_idx = torch.where(signed >= -args.allowed_penetration)[0]
            if exterior_idx.numel() > 0:
                exterior_keep = min(keep, exterior_idx.numel())
                local_order = torch.topk(-unsigned[exterior_idx], exterior_keep, largest=True).indices
                candidate_idx = exterior_idx[local_order]
            else:
                candidate_idx = torch.topk(signed, keep, largest=True).indices

        link_jac = band_jacs[link_name][0].detach().cpu().numpy().astype(np.float64)
        for point_idx in candidate_idx.detach().cpu().tolist():
            point_xyz_t = world_pc[point_idx]
            point_xyz = point_xyz_t.detach().cpu().numpy().astype(np.float64)
            distances = torch.cdist(point_xyz_t[None, None, :], object_pc[None, :, :])[0, 0]
            nearest_idx = int(torch.argmin(distances).detach().cpu())
            normal = normals[nearest_idx].detach().cpu().numpy().astype(np.float64)
            point_jac = point_jacobian_from_link(se3, link_jac, point_xyz)
            phi = float(signed[point_idx].detach().cpu())

            far_slack = cp.Variable(nonneg=True)
            constraints.append(
                phi + normal @ point_jac @ dq - far_slack <= args.contact_band_max_distance
            )
            objective_terms.append(args.contact_band_weight * cp.sum_squares(far_slack))

            if args.contact_band_penetration:
                pen_slack = cp.Variable(nonneg=True)
                constraints.append(
                    phi + normal @ point_jac @ dq + pen_slack >= -args.allowed_penetration
                )
                objective_terms.append(args.collision_slack_weight * cp.sum_squares(pen_slack))



def add_self_collision_constraints(
    hand,
    q: torch.Tensor,
    status,
    dq,
    objective_terms: list,
    constraints: list,
    finger_idx: int,
    finger_mask: list[int],
    args: argparse.Namespace,
) -> None:
    """Keep the moving finger at least self_collision_min_distance from others.

    Only collisions relevant to the moving reduced hand are checked. For a
    moving non-thumb finger, the obstacles are the active thumb and palm. For a
    moving thumb, the obstacles are the active non-thumb fingers and palm.
    """

    if not args.self_collision:
        return

    active_obstacle_fingers: list[int]
    if finger_idx == 0:
        active_obstacle_fingers = [
            idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0
        ]
    else:
        active_obstacle_fingers = [0] if finger_mask[0] and args.self_collision_include_thumb else []

    obstacle_chunks = []
    if active_obstacle_fingers:
        obstacle_chunks.append(
            sampled_finger_points(
                hand,
                q,
                active_obstacle_fingers,
                args.self_collision_points,
            )
        )

    palm_links = [
        name
        for name in getattr(args, "self_collision_palm_links", [])
        if name in status and name in hand.links_pc
    ]
    for palm_link in palm_links:
        local_pc = sample_points(hand.links_pc[palm_link].to(q.device), args.self_collision_points)
        if local_pc.numel() == 0:
            continue
        se3 = status[palm_link].get_matrix()[0].to(q.device)
        ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=q.device)
        obstacle_chunks.append((torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3])

    obstacle_chunks = [chunk for chunk in obstacle_chunks if chunk.numel() > 0]
    if not obstacle_chunks:
        return
    obstacle_points = torch.cat(obstacle_chunks, dim=0)
    if obstacle_points.numel() == 0:
        return

    device = q.device
    links = [name for name in SHADOWHAND_FINGER_LINK_MAP[finger_idx] if name in status and name in hand.links_pc]
    if not links:
        return

    self_jacs = jacobian(hand.pk_chain, q, status, links)
    for link_name in links:
        local_pc = sample_points(hand.links_pc[link_name].to(device), args.self_collision_points)
        if local_pc.numel() == 0:
            continue
        se3 = status[link_name].get_matrix()[0].to(device)
        ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=device)
        world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
        pair_dist = torch.cdist(world_pc[None, :, :], obstacle_points[None, :, :])[0]
        min_dist, nearest_idx = pair_dist.min(dim=1)
        keep = min(args.self_collision_k, min_dist.numel())
        candidate_idx = torch.topk(args.self_collision_margin - min_dist, keep, largest=True).indices

        frame_xyz = se3[:3, 3].detach().cpu().numpy().astype(np.float64)
        link_jac = self_jacs[link_name][0].detach().cpu().numpy().astype(np.float64)
        jv = link_jac[:3]
        jw = link_jac[3:]

        for point_idx in candidate_idx.detach().cpu().tolist():
            dist = float(min_dist[point_idx].detach().cpu())
            if dist > args.self_collision_margin:
                continue
            point_xyz_t = world_pc[point_idx]
            obstacle_xyz_t = obstacle_points[int(nearest_idx[point_idx].detach().cpu())]
            normal_t = (point_xyz_t - obstacle_xyz_t) / (min_dist[point_idx] + 1e-8)
            point_xyz = point_xyz_t.detach().cpu().numpy().astype(np.float64)
            normal = normal_t.detach().cpu().numpy().astype(np.float64)
            offset = point_xyz - frame_xyz
            skew_offset = np.array(
                [
                    [0.0, -offset[2], offset[1]],
                    [offset[2], 0.0, -offset[0]],
                    [-offset[1], offset[0], 0.0],
                ],
                dtype=np.float64,
            )
            point_jac = jv - skew_offset @ jw
            slack = cp.Variable(nonneg=True)
            constraints.append(
                dist + normal @ point_jac @ dq + slack >= args.self_collision_min_distance
            )
            objective_terms.append(args.self_collision_slack_weight * cp.sum_squares(slack))



def closest_surface_point_linearization(
    hand,
    q: torch.Tensor,
    status,
    link_jacs: dict,
    link_name: str,
    patch_target: np.ndarray,
    candidate_name: str,
    max_points: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the target-tracking point y(q) and its Jacobian J_y.

    The selected point can be the hard nearest point to the patch target, a
    softmin average, or the semantic candidate center. The caller uses:

        || y(q) + J_y dq - target ||^2
    """

    if link_name not in status or link_name not in hand.links_pc:
        return None
    device = q.device
    local_pc = sample_points(hand.links_pc[link_name].to(device), max_points)
    if local_pc.numel() == 0:
        return None
    local_pc = subset_for_candidate(link_name, local_pc, candidate_name, args)
    if local_pc.numel() == 0:
        return None

    se3 = status[link_name].get_matrix()[0].to(device)
    ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=device)
    world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
    target_t = torch.as_tensor(patch_target, dtype=q.dtype, device=device)
    distances = torch.linalg.norm(world_pc - target_t[None, :], dim=1)

    if args.target_point_mode == "candidate_center":
        point_xyz_t = world_pc.mean(dim=0)
    elif args.target_point_mode == "softmin":
        weights = torch.softmax(-(distances**2) / max(args.target_softmin_temp, 1e-8), dim=0)
        point_xyz_t = (weights[:, None] * world_pc).sum(dim=0)
    else:
        point_idx = int(torch.argmin(distances).detach().cpu())
        point_xyz_t = world_pc[point_idx]

    frame_xyz = se3[:3, 3].detach().cpu().numpy().astype(np.float64)
    point_xyz = point_xyz_t.detach().cpu().numpy().astype(np.float64)
    link_jac = link_jacs[link_name][0].detach().cpu().numpy().astype(np.float64)
    point_jac = point_jacobian_from_link(se3, link_jac, point_xyz)
    return point_xyz, point_jac
