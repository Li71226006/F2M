from __future__ import annotations

"""Responsibility targets for five-to-fewer-finger F2M optimization.

This module is Part A of the method: it turns a full-hand grasp and the current
reduced-hand grasp into patch-level responsibility distributions, then chooses
target patches that reduce their gap. The key quantities are:

    A[f, k]     finger-to-object-patch affinity
    R_D[k]      responsibility mass left by disabled fingers
    R_res[k]    residual responsibility after active fingers cover some mass
    C*_f        target patch/link/candidate list for one active finger
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from .constants import FINGER_NAMES, SHADOWHAND_FINGER_LINK_MAP
from .geometry import sample_points, signed_distances
from .graph_utils import build_link_patch_graph


FINGER_PREFIXES = {
    "thumb": "THJ",
    "index": "FFJ",
    "middle": "MFJ",
    "ring": "RFJ",
    "little": "LFJ",
}


def sampled_finger_points(hand, q: torch.Tensor, finger_indices: list[int], max_points_per_link: int) -> torch.Tensor:
    robot_pc_dict, _ = hand.get_transformed_links_pc(q)
    chunks = []
    for finger_idx in finger_indices:
        for link_name in SHADOWHAND_FINGER_LINK_MAP[finger_idx]:
            if link_name in robot_pc_dict and robot_pc_dict[link_name].numel() > 0:
                chunks.append(sample_points(robot_pc_dict[link_name], max_points_per_link))
    if not chunks:
        return q.new_zeros((0, 3))
    return torch.cat(chunks, dim=0)



@dataclass(frozen=True)
class FingerTarget:
    finger_idx: int
    link_name: str
    patch_idx: int
    weight: float
    target: np.ndarray
    candidate_name: str = "nearest"
    contact_part: str = "auto"



def link_patch_to_finger_coverage(link_patch: np.ndarray, link_fingers: list[int]) -> np.ndarray:
    """Aggregate link-patch affinity into finger-patch responsibility.

    Formula view:
        A[f, k] = sum_{i in links(f)} A_link[i, k]

    The rows are intentionally not normalized. Large row mass means that the
    finger is geometrically close to more object patches and should carry more
    responsibility in the current pose.
    """

    coverage = np.zeros((len(FINGER_NAMES), link_patch.shape[1]), dtype=np.float64)
    for link_idx, finger_idx in enumerate(link_fingers):
        coverage[finger_idx] += link_patch[link_idx]
    return coverage



def finger_joint_indices(hand, finger_idx: int) -> np.ndarray:
    prefix = FINGER_PREFIXES[FINGER_NAMES[finger_idx]]
    return np.asarray(
        [
            idx
            for idx, name in enumerate(hand.get_joint_orders())
            if name.startswith(prefix)
        ],
        dtype=int,
    )



def allowed_target_links(finger_idx: int, scope: str) -> list[str]:
    links = SHADOWHAND_FINGER_LINK_MAP[finger_idx]
    if scope == "distal":
        return [name for name in links if name.endswith("distal")]
    if scope == "distal_middle":
        return [
            name
            for name in links
            if name.endswith("distal") or name.endswith("middle")
        ]
    return list(links)



def allowed_contact_part_links(finger_idx: int, contact_part: str) -> list[str]:
    links = SHADOWHAND_FINGER_LINK_MAP[finger_idx]
    if contact_part == "distal":
        return [name for name in links if name.endswith("distal")]
    if contact_part == "middle":
        return [name for name in links if name.endswith("middle")]
    if contact_part == "proximal":
        return [
            name
            for name in links
            if name.endswith("proximal") or name.endswith("metacarpal")
        ]
    return list(links)



def semantic_link_tier(link_name: str) -> int:
    if link_name.endswith("distal"):
        return 0
    if link_name.endswith("middle"):
        return 1
    return 2



def semantic_link_tier_name(link_name: str) -> str:
    tier = semantic_link_tier(link_name)
    if tier == 0:
        return "distal"
    if tier == 1:
        return "middle"
    return "other"



def contact_part_prior(contact_part: str, args: argparse.Namespace) -> float:
    priors = {
        "distal": args.part_prior_distal,
        "middle": args.part_prior_middle,
        "proximal": args.part_prior_proximal,
        "auto": 1.0,
    }
    return float(priors.get(contact_part, 1.0))



def contact_candidate_indices(
    link_name: str,
    local_pc: torch.Tensor,
    args: argparse.Namespace,
) -> list[tuple[str, torch.Tensor, float]]:
    """Return local surface candidate regions for a link.

    The old target objective could choose any sampled point on a link. That is
    brittle because the nearest point may lie on a side face or a proximal/root
    part of the link. This helper makes the correspondence slightly more
    anatomical by exposing a few coarse "pad" candidates per link.
    """

    n_points = local_pc.shape[0]
    if n_points == 0:
        return []
    all_idx = torch.arange(n_points, device=local_pc.device)
    if args.target_contact_candidate_mode == "nearest" or n_points < 8:
        return [("nearest", all_idx, 0.0)]

    span = local_pc.max(dim=0).values - local_pc.min(dim=0).values
    axis = int(torch.argmax(span).detach().cpu())
    order = torch.argsort(local_pc[:, axis])
    keep = max(4, int(round(n_points * args.target_candidate_fraction)))
    keep = min(keep, n_points)
    low = order[:keep]
    high = order[-keep:]
    center_start = max(0, n_points // 2 - keep // 2)
    center = order[center_start : center_start + keep]

    if link_name.endswith("distal"):
        return [
            ("distal_end_a", low, 0.05),
            ("distal_end_b", high, 0.05),
            ("distal_center", center, 0.20),
        ]
    if link_name.endswith("middle"):
        return [
            ("middle_center", center, 0.08),
            ("middle_end_a", low, 0.18),
            ("middle_end_b", high, 0.18),
        ]
    if link_name.endswith("proximal") or link_name.endswith("metacarpal"):
        return [
            ("proximal_center", center, 0.35),
            ("proximal_end_a", low, 0.45),
            ("proximal_end_b", high, 0.45),
        ]
    return [("link_center", center, 0.45), ("nearest", all_idx, 0.60)]



def subset_for_candidate(
    link_name: str,
    local_pc: torch.Tensor,
    candidate_name: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    candidates = contact_candidate_indices(link_name, local_pc, args)
    for name, idx, _ in candidates:
        if name == candidate_name:
            return local_pc[idx]
    return local_pc



def ordered_active_nonthumb(finger_mask: list[int]) -> list[int]:
    """Move active non-thumb fingers nearest to the disabled fingers first."""

    disabled = [idx for idx, enabled in enumerate(finger_mask) if not enabled and idx != 0]
    active = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
    if not disabled:
        return active
    return sorted(active, key=lambda idx: (min(abs(idx - d) for d in disabled), idx))



def compute_finger_responsibility(
    hand,
    q: torch.Tensor,
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Compute the geometric responsibility graph at pose q.

    The link graph computes a soft affinity from each hand surface link to each
    object patch:

        A_link[i, k] = exp(-d(i, k)^2 / sigma^2) * direction_factor

    Then link rows are summed into finger rows. This is not a force solver; it
    is the responsibility proxy used to compare full-hand and reduced-hand
    coverage over object patches.
    """

    link_patch, link_names, link_fingers = build_link_patch_graph(
        hand,
        q,
        patch_positions,
        top_k=args.c0_top_k,
        sigma=args.c0_sigma,
        radius=args.link_radius,
        patch_normals=patch_normals,
        points_per_link=args.link_points_per_link,
        use_direction_factor=not args.no_direction_factor,
        use_signed_band=args.responsibility_signed_band,
        outside_band=args.responsibility_outside_band,
        allowed_penetration=args.allowed_penetration,
        tangent_sigma=args.responsibility_tangent_sigma,
        penetration_sigma=args.responsibility_penetration_sigma,
        saturate_contact=args.penetration_saturates_affinity,
        contact_distance=args.affinity_contact_distance,
        contact_radius=args.affinity_contact_radius,
    )
    return link_patch_to_finger_coverage(link_patch, link_fingers), link_patch, link_names, link_fingers



def transformed_link_points(hand, q: torch.Tensor, link_name: str, max_points: int) -> torch.Tensor:
    status = hand.pk_chain.forward_kinematics(q)
    if link_name not in status or link_name not in hand.links_pc:
        return q.new_zeros((0, 3))
    local_pc = sample_points(hand.links_pc[link_name].to(q.device), max_points)
    se3 = status[link_name].get_matrix()[0].to(q.device)
    ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=q.device)
    return (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]



def surface_distances_for_links(
    hand,
    q: torch.Tensor,
    links: list[str],
    patch_positions: np.ndarray,
    max_points: int,
) -> np.ndarray:
    patch_t = torch.as_tensor(patch_positions, dtype=q.dtype, device=q.device)
    distances = np.full((len(links), patch_positions.shape[0]), np.inf, dtype=np.float64)
    for link_idx, link_name in enumerate(links):
        points = transformed_link_points(hand, q, link_name, max_points)
        if points.numel() == 0:
            continue
        dist = torch.cdist(points[None, :, :], patch_t[None, :, :])[0].min(dim=0).values
        distances[link_idx] = dist.detach().cpu().numpy().astype(np.float64)
    return distances



def candidate_distances_for_links(
    hand,
    q: torch.Tensor,
    links: list[str],
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each patch against semantic surface candidates on each link.

    Distance is the main term, but a small candidate priority and normal
    consistency term keep targets on plausible pads instead of arbitrary side
    faces. The returned distance matrix still has shape [link, patch].
    """

    patch_t = torch.as_tensor(patch_positions, dtype=q.dtype, device=q.device)
    normal_t = torch.as_tensor(patch_normals, dtype=q.dtype, device=q.device)
    distances = np.full((len(links), patch_positions.shape[0]), np.inf, dtype=np.float64)
    candidate_names = np.full((len(links), patch_positions.shape[0]), "nearest", dtype=object)
    status = hand.pk_chain.forward_kinematics(q)

    for link_idx, link_name in enumerate(links):
        if link_name not in status or link_name not in hand.links_pc:
            continue
        local_pc = sample_points(hand.links_pc[link_name].to(q.device), args.target_surface_points)
        if local_pc.numel() == 0:
            continue
        se3 = status[link_name].get_matrix()[0].to(q.device)
        best_cost = torch.full((patch_t.shape[0],), float("inf"), dtype=q.dtype, device=q.device)
        best_distance = torch.full_like(best_cost, float("inf"))
        best_names = ["nearest"] * patch_t.shape[0]

        for name, idx, priority in contact_candidate_indices(link_name, local_pc, args):
            cand_local = local_pc[idx]
            ones = torch.ones(cand_local.shape[0], 1, dtype=cand_local.dtype, device=q.device)
            cand_world = (torch.cat([cand_local, ones], dim=1) @ se3.T)[:, :3]
            dmat = torch.cdist(cand_world[None, :, :], patch_t[None, :, :])[0]
            min_dist, nearest_idx = dmat.min(dim=0)
            nearest = cand_world[nearest_idx]
            direction = nearest - patch_t
            direction = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-8)
            outside = torch.clamp(1.0 - (direction * normal_t).sum(dim=1), min=0.0)
            cost = (
                min_dist
                + args.target_candidate_priority_weight * priority
                + args.target_normal_weight * outside
            )
            improved = cost < best_cost
            best_cost = torch.where(improved, cost, best_cost)
            best_distance = torch.where(improved, min_dist, best_distance)
            for patch_idx in torch.where(improved)[0].detach().cpu().tolist():
                best_names[patch_idx] = name

        distances[link_idx] = best_distance.detach().cpu().numpy().astype(np.float64)
        candidate_names[link_idx] = np.asarray(best_names, dtype=object)
    return distances, candidate_names



def target_scoring_distance(
    links: list[str],
    distances: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Prefer distal pads when deciding which residual patches a finger should take."""

    if args.target_link_priority_mode == "distance":
        return distances.min(axis=0)

    distal = [idx for idx, name in enumerate(links) if semantic_link_tier(name) == 0]
    if not distal:
        return distances.min(axis=0)

    all_min = distances.min(axis=0)
    distal_min = distances[distal].min(axis=0)
    if not args.target_middle_fallback:
        return distal_min

    use_distal = distal_min <= args.target_distal_max_distance
    return np.where(use_distal, distal_min, all_min)



def choose_semantic_target_link(
    links: list[str],
    distances: np.ndarray,
    patch_idx: int,
    args: argparse.Namespace,
) -> int:
    """Choose the link that should execute a target patch.

    Distance-only matching often lets middle/proximal links steal targets that
    should be finger-pad contacts. The default policy is therefore distal-first,
    with middle links used only as a reachability fallback.
    """

    if args.target_link_priority_mode == "distance":
        return int(np.argmin(distances[:, patch_idx]))

    for tier in (0, 1, 2):
        if tier == 1 and not args.target_middle_fallback:
            continue
        candidates = [
            idx
            for idx, name in enumerate(links)
            if semantic_link_tier(name) == tier and np.isfinite(distances[idx, patch_idx])
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: distances[idx, patch_idx])
        if tier == 0 and distances[best, patch_idx] > args.target_distal_max_distance:
            continue
        return int(best)

    return int(np.argmin(distances[:, patch_idx]))



def select_targets_for_finger(
    hand,
    q: torch.Tensor,
    finger_idx: int,
    residual: np.ndarray,
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args: argparse.Namespace,
    contact_part: str | None = None,
) -> list[FingerTarget]:
    """Convert residual responsibility into QP targets for one finger.

    Abstract score:

        score[k] = R_res[k] * S_reach(f, k) * S_part(part)

    After scoring, non-maximum suppression keeps targets spatially separated,
    and each chosen patch is assigned to a concrete link/candidate surface
    point. The QP later tracks target = patch_center + target_gap * normal.
    """

    status = hand.pk_chain.forward_kinematics(q)
    if contact_part:
        links = [name for name in allowed_contact_part_links(finger_idx, contact_part) if name in status]
    else:
        links = [name for name in allowed_target_links(finger_idx, args.target_link_scope) if name in status]
    if not links:
        return []

    if args.target_contact_candidate_mode == "nearest":
        dist = surface_distances_for_links(
            hand,
            q,
            links,
            patch_positions,
            args.target_surface_points,
        )
        candidate_names = np.full(dist.shape, "nearest", dtype=object)
    else:
        dist, candidate_names = candidate_distances_for_links(
            hand,
            q,
            links,
            patch_positions,
            patch_normals,
            args,
        )
    if contact_part:
        scoring_dist = dist.min(axis=0)
    else:
        scoring_dist = target_scoring_distance(links, dist, args)
    reach = np.exp(-(scoring_dist**2) / (args.target_reach_sigma**2))
    scores = contact_part_prior(contact_part or "auto", args) * np.maximum(0.0, residual) * reach

    selected: list[int] = []
    suppressed = np.zeros_like(scores, dtype=bool)
    for _ in range(args.targets_per_finger):
        current = scores.copy()
        current[suppressed] = -1.0
        patch_idx = int(np.argmax(current))
        if current[patch_idx] <= args.min_target_weight:
            break
        selected.append(patch_idx)
        patch_dist = np.linalg.norm(patch_positions - patch_positions[patch_idx], axis=1)
        suppressed |= patch_dist < args.nms_radius

    targets: list[FingerTarget] = []
    for patch_idx in selected:
        if contact_part:
            link_idx = int(np.argmin(dist[:, patch_idx]))
        else:
            link_idx = choose_semantic_target_link(links, dist, patch_idx, args)
        targets.append(
            FingerTarget(
                finger_idx=finger_idx,
                link_name=links[link_idx],
                patch_idx=patch_idx,
                weight=float(scores[patch_idx]),
                target=patch_positions[patch_idx] + args.target_gap * patch_normals[patch_idx],
                candidate_name=str(candidate_names[link_idx, patch_idx]),
                contact_part=contact_part or semantic_link_tier_name(links[link_idx]),
            )
        )
    return targets



def active_contact_centroid(
    hand,
    q: torch.Tensor,
    finger_mask: list[int],
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray | None:
    active_nonthumb = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
    if not active_nonthumb:
        return None
    points = sampled_finger_points(hand, q, active_nonthumb, args.stats_points_per_link)
    if points.numel() == 0:
        return None
    object_pc = object_pc_normals[:, :3].to(q.device)
    normals = object_pc_normals[:, 3:].to(q.device)
    _, unsigned = signed_distances(points, object_pc, normals)
    near = unsigned <= args.thumb_opposition_near_distance
    if int(near.sum().detach().cpu()) > 0:
        chosen = points[near]
    else:
        keep = min(args.thumb_opposition_fallback_points, points.shape[0])
        idx = torch.topk(-unsigned, keep, largest=True).indices
        chosen = points[idx]
    return chosen.mean(dim=0).detach().cpu().numpy().astype(np.float64)



def patch_wrench_matrix(patch_positions: np.ndarray, patch_normals: np.ndarray) -> np.ndarray:
    """Build a low-cost patch wrench proxy [force, torque].

    The assumed force direction is inward along -normal. This is not a friction
    cone or force-closure model; it is only used for balancing heuristics.
    """

    object_center = patch_positions.mean(axis=0)
    inward = -patch_normals
    torque = np.cross(patch_positions - object_center[None, :], inward)
    return np.concatenate([inward, torque], axis=1)



def select_thumb_targets(
    hand,
    q: torch.Tensor,
    finger_mask: list[int],
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> list[FingerTarget]:
    """Choose thumb targets on the object side opposite active-finger support.

    The thumb is deliberately separate from residual responsibility transfer.
    It is scored for opposition/support and wrench balancing, rather than being
    forced to chase the patches left by disabled non-thumb fingers.
    """

    from .metrics import wrench_balance_stats

    if not args.thumb_contact_targets:
        return []
    status = hand.pk_chain.forward_kinematics(q)
    links = [name for name in allowed_target_links(0, args.thumb_target_link_scope) if name in status]
    if not links:
        return []

    object_center = patch_positions.mean(axis=0)
    active_centroid = active_contact_centroid(hand, q, finger_mask, object_pc_normals, args)
    if active_centroid is None:
        thumb_points = sampled_finger_points(hand, q, [0], args.stats_points_per_link)
        if thumb_points.numel() == 0:
            return []
        active_centroid = thumb_points.mean(dim=0).detach().cpu().numpy().astype(np.float64)

    support_dir = active_centroid - object_center
    support_norm = np.linalg.norm(support_dir)
    if support_norm < 1e-8:
        return []
    desired_dir = -support_dir / support_norm

    patch_vec = patch_positions - object_center[None, :]
    patch_vec = patch_vec / (np.linalg.norm(patch_vec, axis=1, keepdims=True) + 1e-8)
    opposite_score = np.maximum(0.0, patch_vec @ desired_dir)
    normal_score = np.maximum(0.0, patch_normals @ desired_dir)
    current_wrench = wrench_balance_stats(
        hand,
        q,
        finger_mask,
        patch_positions,
        patch_normals,
        args,
        include_thumb=True,
    )
    patch_wrenches = patch_wrench_matrix(patch_positions, patch_normals)
    net_hint = np.zeros(6, dtype=np.float64)
    if current_wrench["wrench_norm"] > 1e-12:
        coverage, _, _, _ = compute_finger_responsibility(
            hand,
            q,
            patch_positions,
            patch_normals,
            args,
        )
        rows = [idx for idx, enabled in enumerate(finger_mask) if enabled]
        weights = coverage[rows].sum(axis=0) if rows else np.zeros(patch_positions.shape[0])
        if weights.sum() > 1e-12:
            net_hint = (weights[:, None] * patch_wrenches).sum(axis=0) / (weights.sum() + 1e-12)
    wrench_score = np.maximum(0.0, -(patch_wrenches @ net_hint) / (np.linalg.norm(net_hint) + 1e-8))

    if args.target_contact_candidate_mode == "nearest":
        dist = surface_distances_for_links(
            hand,
            q,
            links,
            patch_positions,
            args.target_surface_points,
        )
        candidate_names = np.full(dist.shape, "nearest", dtype=object)
    else:
        dist, candidate_names = candidate_distances_for_links(
            hand,
            q,
            links,
            patch_positions,
            patch_normals,
            args,
        )

    reach = np.exp(-(dist.min(axis=0) ** 2) / (args.thumb_reach_sigma**2))
    scores = (
        args.thumb_opposition_weight * opposite_score
        + args.thumb_normal_weight * normal_score
        + args.thumb_wrench_weight * wrench_score
    ) * reach

    targets: list[FingerTarget] = []
    suppressed = np.zeros_like(scores, dtype=bool)
    for _ in range(args.thumb_targets):
        current = scores.copy()
        current[suppressed] = -1.0
        patch_idx = int(np.argmax(current))
        if current[patch_idx] <= args.min_target_weight:
            break
        link_idx = int(np.argmin(dist[:, patch_idx]))
        targets.append(
            FingerTarget(
                finger_idx=0,
                link_name=links[link_idx],
                patch_idx=patch_idx,
                weight=float(args.thumb_target_weight * current[patch_idx]),
                target=patch_positions[patch_idx] + args.target_gap * patch_normals[patch_idx],
                candidate_name=str(candidate_names[link_idx, patch_idx]),
            )
        )
        patch_dist = np.linalg.norm(patch_positions - patch_positions[patch_idx], axis=1)
        suppressed |= patch_dist < args.nms_radius
    return targets
