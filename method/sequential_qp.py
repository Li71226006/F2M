from __future__ import annotations

"""Sequential QP optimizer for the F2M/CFET method.

This module is Part B of the method. It repeatedly compares the disabled
full-hand responsibility R_D with the current reduced-hand responsibility
A^r(q), then updates q to shrink that responsibility gap:

    active non-thumb fingers -> thumb opposition -> palm/wrist prealignment

Each QP is local: nonlinear nearest-point, SDF, and Jacobian quantities are
linearized at the current q, solved, accepted/rejected, then recomputed.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import torch

from .config import CFETConfig
from .constants import FINGER_NAMES
from .contact_targets import (
    FingerTarget,
    active_contact_centroid,
    allowed_target_links,
    allowed_contact_part_links,
    compute_finger_responsibility,
    finger_joint_indices,
    ordered_active_nonthumb,
    patch_wrench_matrix,
    select_targets_for_finger,
    select_thumb_targets,
    subset_for_candidate,
)
from .func import output_root, save_json
from .geometry import sample_points, signed_distances
from .graph_utils import write_cstar_csv
from .kinematics import jacobian
from .metrics import (
    object_gap_stats,
    reduced_penetration_stats,
    single_finger_gap_stats,
    thumb_opposition_stats,
    wrench_balance_stats,
)
from .object_processor import ObjectProcessor
from .qp_terms import (
    add_contact_band_constraints,
    add_self_collision_constraints,
    add_surface_collision_constraints,
    closest_surface_point_linearization,
    point_jacobian_from_link,
)
from .responsibility import ResponsibilityComputer
from .responsibility_gap import (
    active_nonthumb_rows,
    build_patch_compensation_kernel,
    compensated_coverage,
    finite_difference_finger_responsibility_jacobian,
    positive_gap,
    raw_self_retention_ratio,
    target_responsibility,
)
from .robot_processor import RobotProcessor
from visualizer.mujoco_renderer import MujocoRenderer

def solve_single_finger_qp(
    hand,
    q: torch.Tensor,
    q_ref: torch.Tensor,
    finger_idx: int,
    finger_mask: list[int],
    targets: list[FingerTarget],
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    """Solve one local QP for a single moving finger.

    Variables:
        dq in R^n, with all non-finger joints frozen.

    Main objective:
        sum_m w_m || y_m(q) + J_m dq - z_m ||^2
        + joint_anchor + step_regularization

    Safety/geometry terms are appended by qp_terms.py: object penetration,
    near-surface band, and self-collision.
    """

    if not targets and not args.contact_band:
        return q, {"finger": FINGER_NAMES[finger_idx], "num_targets": 0, "status": "no_targets"}

    device = q.device
    n = q.numel()
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    q_current = q.detach().clone()
    finger_joints = finger_joint_indices(hand, finger_idx)
    frozen = np.ones(n, dtype=bool)
    frozen[finger_joints] = False

    last_status = None
    last_value = None
    for _ in range(args.qp_iters):
        status = hand.pk_chain.forward_kinematics(q_current)
        link_names = sorted({target.link_name for target in targets})
        jacs = jacobian(hand.pk_chain, q_current, status, link_names)
        q_np = q_current.detach().cpu().numpy().astype(np.float64)
        q_ref_np = q_ref.detach().cpu().numpy().astype(np.float64)

        dq = cp.Variable(n)
        # Full-size dq keeps Jacobian code simple; equality constraints freeze
        # every joint outside the current finger.
        objective_terms = [args.step_weight * cp.sum_squares(dq)]
        constraints = [
            dq[frozen] == 0.0,
            dq[finger_joints] <= np.minimum(args.max_step, upper[finger_joints] - q_np[finger_joints]),
            dq[finger_joints] >= np.maximum(-args.max_step, lower[finger_joints] - q_np[finger_joints]),
        ]

        for target in targets:
            linearized = closest_surface_point_linearization(
                hand,
                q_current,
                status,
                jacs,
                target.link_name,
                target.target,
                target.candidate_name,
                args.target_surface_points,
                args,
            )
            if linearized is None:
                continue
            point_xyz, point_jac = linearized
            weight = args.contact_weight * max(target.weight, args.min_target_weight)
            objective_terms.append(weight * cp.sum_squares(point_xyz + point_jac @ dq - target.target))

        objective_terms.append(
            args.joint_anchor_weight
            * cp.sum_squares(q_np[finger_joints] + dq[finger_joints] - q_ref_np[finger_joints])
        )
        add_surface_collision_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            object_pc_normals,
            args,
        )
        add_contact_band_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            object_pc_normals,
            args,
        )
        add_self_collision_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            finger_mask,
            args,
        )

        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            problem.solve(
                solver="OSQP",
                warm_start=True,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=args.osqp_max_iter,
            )
        except Exception:
            problem.solve(solver="CLARABEL", verbose=False)
        last_status = problem.status
        last_value = float(problem.value) if problem.value is not None else None
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT} or dq.value is None:
            break

        delta = torch.as_tensor(dq.value, dtype=q_current.dtype, device=device)
        q_current = q_current + args.line_search * delta
        if float(delta[finger_joints].norm().detach().cpu()) < args.converge_delta:
            break

    return q_current.detach(), {
        "finger": FINGER_NAMES[finger_idx],
        "num_targets": len(targets),
        "status": last_status,
        "last_objective": last_value,
        "target_patches": [target.patch_idx for target in targets],
        "target_links": [target.link_name for target in targets],
        "target_candidates": [target.candidate_name for target in targets],
        "target_contact_parts": [target.contact_part for target in targets],
    }

def solve_single_finger_responsibility_gap_qp(
    hand,
    q: torch.Tensor,
    q_ref: torch.Tensor,
    q_full: torch.Tensor,
    finger_idx: int,
    finger_mask: list[int],
    disabled_responsibility: np.ndarray,
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    patch_kernel: np.ndarray,
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    """Move one finger in the direction that reduces responsibility gap.

    This is the v2 active-finger objective. It does not pre-assign a patch to a
    link. Instead, it finite-differences the moving finger's responsibility map
    and solves a local QP:

        min || positive_part(T - K^T R_active(q + dq_f)) ||^2

    plus self-retention, a wrench-balance proxy, anchor, step, and safety terms.
    """

    device = q.device
    n = q.numel()
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    q_current = q.detach().clone()
    q_ref_np = q_ref.detach().cpu().numpy().astype(np.float64)

    full_c, _, _, _ = compute_finger_responsibility(hand, q_full, patch_positions, patch_normals, args)
    target = target_responsibility(
        full_c,
        disabled_responsibility,
        finger_mask,
        disabled_weight=args.disabled_compensation_weight,
    )
    active_rows = active_nonthumb_rows(finger_mask)
    finger_joints = finger_joint_indices(hand, finger_idx)
    frozen = np.ones(n, dtype=bool)
    frozen[finger_joints] = False

    last_status = None
    last_value = None
    last_gap_before = None
    last_gap_linearized = None
    last_self_before = None
    last_direction_targets = 0
    last_direction_target_links: list[str] = []
    last_direction_target_candidates: list[str] = []
    last_direction_target_parts: list[str] = []
    last_direction_target_region_sizes: list[int] = []
    last_wrench_before = None
    last_wrench_linearized = None
    patch_wrenches = patch_wrench_matrix(patch_positions, patch_normals)
    for _ in range(args.qp_iters):
        current_c, _, _, _ = compute_finger_responsibility(hand, q_current, patch_positions, patch_normals, args)
        active_total = current_c[active_rows].sum(axis=0) if active_rows else np.zeros(patch_positions.shape[0])
        base_comp = compensated_coverage(active_total, patch_kernel)
        gap_before = positive_gap(target, active_total, patch_kernel)
        base_finger, jac_finger, fd_joints = finite_difference_finger_responsibility_jacobian(
            hand,
            q_current,
            finger_idx,
            patch_positions,
            patch_normals,
            args,
            eps=args.responsibility_fd_eps,
        )
        if fd_joints.size == 0:
            return q_current.detach(), {
                "finger": FINGER_NAMES[finger_idx],
                "phase": "active_finger",
                "status": "no_finger_joints",
                "objective_mode": "responsibility_gap",
            }
        gap_jac = patch_kernel.T @ jac_finger
        self_base = base_finger
        self_jac = jac_finger
        self_target = full_c[finger_idx].astype(np.float64)
        self_before = raw_self_retention_ratio(base_finger, self_target)

        if args.responsibility_wrench_weight > 0.0:
            wrench_rows = [idx for idx, enabled in enumerate(finger_mask) if enabled]
            wrench_weights = (
                current_c[wrench_rows].sum(axis=0)
                if wrench_rows
                else np.zeros(patch_positions.shape[0], dtype=np.float64)
            )
            wrench_mass = max(float(wrench_weights.sum()), 1e-8)
            base_wrench = (wrench_weights @ patch_wrenches) / wrench_mass
            wrench_jac = (patch_wrenches.T @ jac_finger) / wrench_mass
            wrench_scale = np.asarray(
                [1.0, 1.0, 1.0, args.responsibility_wrench_torque_scale,
                 args.responsibility_wrench_torque_scale, args.responsibility_wrench_torque_scale],
                dtype=np.float64,
            )
        else:
            base_wrench = None
            wrench_jac = None
            wrench_scale = None

        status = hand.pk_chain.forward_kinematics(q_current)
        q_np = q_current.detach().cpu().numpy().astype(np.float64)
        dq = cp.Variable(n)
        gap_slack = cp.Variable(patch_positions.shape[0], nonneg=True)
        self_slack = cp.Variable(patch_positions.shape[0], nonneg=True)
        linear_comp = base_comp + gap_jac @ dq[fd_joints]
        linear_self = self_base + self_jac @ dq[fd_joints]
        objective_terms = [
            args.responsibility_gap_weight * cp.sum_squares(gap_slack),
            args.responsibility_self_weight * cp.sum_squares(self_slack),
            args.step_weight * cp.sum_squares(dq),
            args.joint_anchor_weight
            * cp.sum_squares(q_np[finger_joints] + dq[finger_joints] - q_ref_np[finger_joints]),
        ]
        constraints = [
            dq[frozen] == 0.0,
            dq[finger_joints] <= np.minimum(args.max_step, upper[finger_joints] - q_np[finger_joints]),
            dq[finger_joints] >= np.maximum(-args.max_step, lower[finger_joints] - q_np[finger_joints]),
            gap_slack >= target - linear_comp,
            self_slack >= self_target - linear_self,
        ]
        if base_wrench is not None and wrench_jac is not None and wrench_scale is not None:
            linear_wrench = base_wrench + wrench_jac @ dq[fd_joints]
            objective_terms.append(
                args.responsibility_wrench_weight
                * cp.sum_squares(cp.multiply(wrench_scale, linear_wrench))
            )
            last_wrench_before = float(np.linalg.norm(wrench_scale * base_wrench))
        direction_targets = []
        if args.responsibility_direction_weight > 0.0:
            direction_targets = select_targets_for_finger(
                hand,
                q_current,
                finger_idx,
                gap_before,
                patch_positions,
                patch_normals,
                args,
            )[: args.responsibility_direction_targets]
            if direction_targets:
                link_names = sorted({target.link_name for target in direction_targets})
                direction_jacs = jacobian(hand.pk_chain, q_current, status, link_names)
                for direction_target in direction_targets:
                    linearized = closest_surface_point_linearization(
                        hand,
                        q_current,
                        status,
                        direction_jacs,
                        direction_target.link_name,
                        direction_target.target,
                        direction_target.candidate_name,
                        args.target_surface_points,
                        args,
                    )
                    if linearized is None:
                        continue
                    point_xyz, point_jac = linearized
                    weight = args.responsibility_direction_weight * max(
                        direction_target.weight,
                        args.min_target_weight,
                    )
                    objective_terms.append(
                        weight * cp.sum_squares(point_xyz + point_jac @ dq - direction_target.target)
                    )
        last_direction_targets = len(direction_targets)
        last_direction_target_links = [target.link_name for target in direction_targets]
        last_direction_target_candidates = [target.candidate_name for target in direction_targets]
        last_direction_target_parts = [target.contact_part for target in direction_targets]
        last_direction_target_region_sizes = [target.region_size for target in direction_targets]
        add_surface_collision_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            object_pc_normals,
            args,
        )
        add_contact_band_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            object_pc_normals,
            args,
        )
        add_self_collision_constraints(
            hand,
            q_current,
            status,
            dq,
            objective_terms,
            constraints,
            finger_idx,
            finger_mask,
            args,
        )

        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            problem.solve(
                solver="OSQP",
                warm_start=True,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=args.osqp_max_iter,
            )
        except Exception:
            problem.solve(solver="CLARABEL", verbose=False)
        last_status = problem.status
        last_value = float(problem.value) if problem.value is not None else None
        last_gap_before = float(np.sum(gap_before))
        last_self_before = self_before
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT} or dq.value is None:
            break

        delta_np = np.asarray(dq.value, dtype=np.float64)
        predicted_comp = base_comp + gap_jac @ delta_np[fd_joints]
        last_gap_linearized = float(np.maximum(0.0, target - predicted_comp).sum())
        if base_wrench is not None and wrench_jac is not None and wrench_scale is not None:
            predicted_wrench = base_wrench + wrench_jac @ delta_np[fd_joints]
            last_wrench_linearized = float(np.linalg.norm(wrench_scale * predicted_wrench))
        delta = torch.as_tensor(delta_np, dtype=q_current.dtype, device=device)
        q_current = q_current + args.line_search * delta
        if float(delta[finger_joints].norm().detach().cpu()) < args.converge_delta:
            break

    final_c, _, _, _ = compute_finger_responsibility(hand, q_current, patch_positions, patch_normals, args)
    final_self = raw_self_retention_ratio(final_c[finger_idx], full_c[finger_idx])
    return q_current.detach(), {
        "finger": FINGER_NAMES[finger_idx],
        "num_targets": 0,
        "num_direction_targets": last_direction_targets,
        "status": last_status,
        "last_objective": last_value,
        "objective_mode": "responsibility_gap",
        "gap_before_sum": last_gap_before,
        "gap_predicted_sum": last_gap_linearized,
        "self_retention_before": last_self_before,
        "self_retention_after": final_self,
        "direction_target_links": last_direction_target_links,
        "direction_target_candidates": last_direction_target_candidates,
        "direction_target_parts": last_direction_target_parts,
        "direction_target_region_sizes": last_direction_target_region_sizes,
        "wrench_before_scaled": last_wrench_before,
        "wrench_predicted_scaled": last_wrench_linearized,
    }

def palm_joint_indices(hand, include_wrist: bool = False) -> np.ndarray:
    """Return virtual palm joints, optionally including physical wrist joints."""

    names = hand.pk_chain.get_joint_parameter_names()
    prefixes = ("virtual_joint_",)
    if include_wrist:
        prefixes = ("virtual_joint_", "WRJ")
    return np.asarray(
        [idx for idx, name in enumerate(names) if name.startswith(prefixes)],
        dtype=int,
    )

def accept_candidate_step(
    before_residual: float,
    after_residual: float,
    before_stats: dict,
    after_stats: dict,
    args: argparse.Namespace,
    before_finger_gap: dict | None = None,
    after_finger_gap: dict | None = None,
) -> bool:
    """Guard active-finger QP output before committing it to the sequence.

    A mathematically feasible QP step can still be a bad grasp update. This
    guard requires either residual/near/penetration improvement globally, or a
    clearly better moving-finger gap without unacceptable global regressions.
    """

    if not args.acceptance_guard:
        return True
    residual_gain = before_residual - after_residual
    near_gain = after_stats["near_ratio"] - before_stats["near_ratio"]
    penetration_worsen = after_stats["max_penetration_mm"] - before_stats["max_penetration_mm"]
    penetration_guard_ok = accept_penetration_guard(
        before_stats,
        after_stats,
        args.accept_max_penetration_worsen_mm,
        args,
    )
    residual_ok = (
        residual_gain >= args.accept_residual_gain
        and penetration_guard_ok
        and near_gain >= -args.accept_near_drop
    )
    near_ok = (
        near_gain >= args.accept_near_gain
        and penetration_guard_ok
    )
    penetration_ok = (
        penetration_worsen <= -args.accept_penetration_gain_mm
        and near_gain >= -args.accept_near_drop
    )
    global_ok = residual_ok or near_ok or penetration_ok
    if not args.moving_finger_acceptance or before_finger_gap is None or after_finger_gap is None:
        return global_ok
    if not before_finger_gap or not after_finger_gap:
        return global_ok

    before_min = before_finger_gap.get("min_unsigned_mm", float("inf"))
    after_min = after_finger_gap.get("min_unsigned_mm", float("inf"))
    before_near = before_finger_gap.get("near_ratio", 0.0)
    after_near = after_finger_gap.get("near_ratio", 0.0)
    finger_distance_gain = before_min - after_min
    finger_near_gain = after_near - before_near
    finger_ok = (
        finger_distance_gain >= args.accept_finger_distance_gain_mm
        or finger_near_gain >= args.accept_finger_near_gain
        or (
            after_min <= args.accept_finger_max_distance_mm
            and after_min <= before_min + args.accept_finger_max_distance_worsen_mm
        )
    )
    finger_dominant_ok = (
        finger_distance_gain >= args.accept_finger_distance_gain_mm
        and penetration_guard_ok
        and near_gain >= -args.accept_contact_near_drop
        and after_residual <= before_residual + args.accept_residual_worsen
    )
    return (global_ok and finger_ok) or finger_dominant_ok

def accept_penetration_guard(
    before_stats: dict,
    after_stats: dict,
    max_worsen_mm: float,
    args: argparse.Namespace,
) -> bool:
    """Reject steps that worsen already excessive hand-object penetration."""

    before_max = float(before_stats["max_penetration_mm"])
    after_max = float(after_stats["max_penetration_mm"])
    limit = float(args.accept_max_total_penetration_mm)
    if after_max <= limit:
        return after_max - before_max <= max_worsen_mm
    if before_max > limit:
        return after_max <= before_max + args.accept_penetration_over_limit_worsen_mm
    return False

def accept_thumb_step(
    before_stats: dict,
    after_stats: dict,
    before_finger_gap: dict,
    after_finger_gap: dict,
    before_opposition: dict,
    after_opposition: dict,
    before_wrench: dict,
    after_wrench: dict,
    args: argparse.Namespace,
) -> bool:
    """Accept thumb steps by contact/opposition/wrench criteria, not residual.

    The thumb is support/opposition. It should not be forced to reduce R_res,
    because R_res is intentionally defined over active non-thumb compensation.
    """

    penetration_worsen = after_stats["max_penetration_mm"] - before_stats["max_penetration_mm"]
    before_min = before_finger_gap.get("min_unsigned_mm", float("inf")) if before_finger_gap else float("inf")
    after_min = after_finger_gap.get("min_unsigned_mm", float("inf")) if after_finger_gap else float("inf")
    distance_gain = before_min - after_min
    near_gain = (
        after_finger_gap.get("near_ratio", 0.0) - before_finger_gap.get("near_ratio", 0.0)
        if before_finger_gap and after_finger_gap
        else 0.0
    )
    opposition_gain = after_opposition["alignment"] - before_opposition["alignment"]
    wrench_gain = before_wrench["wrench_norm"] - after_wrench["wrench_norm"]
    contact_ok = (
        distance_gain >= args.thumb_accept_distance_gain_mm
        or near_gain >= args.thumb_accept_near_gain
        or (
            after_min <= args.thumb_accept_max_distance_mm
            and after_min <= before_min + args.thumb_accept_max_distance_worsen_mm
        )
    )
    opposition_ok = opposition_gain >= args.thumb_accept_opposition_gain
    wrench_ok = wrench_gain >= args.thumb_accept_wrench_gain
    penetration_ok = accept_penetration_guard(
        before_stats,
        after_stats,
        args.thumb_accept_max_penetration_worsen_mm,
        args,
    )
    return penetration_ok and (contact_ok or opposition_ok or wrench_ok)

def accept_palm_step(
    before_stats: dict,
    after_stats: dict,
    before_wrench: dict,
    after_wrench: dict,
    args: argparse.Namespace,
) -> bool:
    """Accept palm/wrist prealignment if reach or wrench improves safely."""

    penetration_worsen = after_stats["max_penetration_mm"] - before_stats["max_penetration_mm"]
    near_gain = after_stats["near_ratio"] - before_stats["near_ratio"]
    wrench_gain = before_wrench["wrench_norm"] - after_wrench["wrench_norm"]
    penetration_ok = accept_penetration_guard(
        before_stats,
        after_stats,
        args.palm_accept_max_penetration_worsen_mm,
        args,
    )
    return penetration_ok and (
        near_gain >= args.palm_accept_near_gain
        or wrench_gain >= args.palm_accept_wrench_gain
    )

def reduced_residual_mass(
    hand,
    q: torch.Tensor,
    disabled_responsibility: np.ndarray,
    finger_mask: list[int],
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args: argparse.Namespace,
) -> float:
    """Compute sum_k max(0, R_D[k] - R_active_non_thumb[k])."""

    coverage, _, _, _ = compute_finger_responsibility(
        hand,
        q,
        patch_positions,
        patch_normals,
        args,
    )
    active_rows = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
    if not active_rows:
        return float(disabled_responsibility.sum())
    active_total = coverage[active_rows].sum(axis=0)
    return float(np.maximum(0.0, disabled_responsibility - active_total).sum())

def responsibility_consistency_stats(
    residual_mass: float,
    disabled_responsibility: np.ndarray,
) -> dict:
    """Return a compact coverage ratio for logs and stats.json."""

    disabled_mass = float(np.sum(disabled_responsibility))
    if disabled_mass <= 1e-12:
        return {
            "disabled_mass": disabled_mass,
            "residual_mass": residual_mass,
            "coverage_ratio": 1.0,
        }
    return {
        "disabled_mass": disabled_mass,
        "residual_mass": residual_mass,
        "coverage_ratio": float(1.0 - residual_mass / (disabled_mass + 1e-12)),
    }

def hypothesis_score(
    hand,
    q_before: torch.Tensor,
    q_candidate: torch.Tensor,
    finger_idx: int,
    before_residual: float,
    after_residual: float,
    before_stats: dict,
    after_stats: dict,
    before_finger_gap: dict,
    after_finger_gap: dict,
    before_wrench: dict | None,
    after_wrench: dict | None,
    accepted: bool,
    args: argparse.Namespace,
) -> float:
    """Rank distal/middle/proximal rollout candidates for one active finger."""

    residual_gain = before_residual - after_residual
    near_gain = after_stats["near_ratio"] - before_stats["near_ratio"]
    penetration_worsen = max(0.0, after_stats["max_penetration_mm"] - before_stats["max_penetration_mm"])
    before_min = before_finger_gap.get("min_unsigned_mm", float("inf")) if before_finger_gap else float("inf")
    after_min = after_finger_gap.get("min_unsigned_mm", float("inf")) if after_finger_gap else float("inf")
    finger_distance_gain = 0.0 if not np.isfinite(before_min + after_min) else before_min - after_min
    finger_near_gain = (
        after_finger_gap.get("near_ratio", 0.0) - before_finger_gap.get("near_ratio", 0.0)
        if before_finger_gap and after_finger_gap
        else 0.0
    )
    wrench_gain = 0.0
    if before_wrench and after_wrench:
        wrench_gain = before_wrench.get("wrench_norm", 0.0) - after_wrench.get("wrench_norm", 0.0)
    finger_joints = finger_joint_indices(hand, finger_idx)
    joint_delta_deg = float(
        torch.linalg.norm((q_candidate - q_before)[finger_joints]).detach().cpu() * 180.0 / math.pi
    )
    score = (
        args.hypothesis_residual_weight * residual_gain
        + args.hypothesis_near_weight * near_gain
        + args.hypothesis_finger_distance_weight * finger_distance_gain
        + args.hypothesis_finger_near_weight * finger_near_gain
        + args.hypothesis_wrench_weight * wrench_gain
        - args.hypothesis_penetration_weight * penetration_worsen
        - args.hypothesis_joint_delta_weight * joint_delta_deg
    )
    if not accepted:
        score -= args.hypothesis_reject_penalty
    return float(score)

def solve_palm_qp(
    hand,
    q: torch.Tensor,
    q_ref: torch.Tensor,
    finger_mask: list[int],
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    """Solve the palm/wrist reach-prealignment QP.

    Variables are the virtual palm pose joints, plus wrist joints if enabled.
    Unlike active-finger QPs, this phase does not chase residual patches; it
    pulls active-link surface samples toward nearby object surface points so the
    whole hand becomes more reachable.
    """

    if not args.palm_phase:
        return q, {"phase": "palm", "status": "disabled"}

    device = q.device
    n = q.numel()
    block_joints = palm_joint_indices(hand, include_wrist=args.palm_phase_include_wrist)
    if block_joints.size == 0:
        return q, {"phase": "palm", "status": "no_palm_joints"}

    lower, upper = hand.pk_chain.get_joint_limits()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    q_current = q.detach().clone()
    q_ref_np = q_ref.detach().cpu().numpy().astype(np.float64)

    last_status = None
    last_value = None
    for _ in range(args.palm_qp_iters):
        status = hand.pk_chain.forward_kinematics(q_current)
        q_np = q_current.detach().cpu().numpy().astype(np.float64)
        dq = cp.Variable(n)
        # Palm phase freezes all finger joints and only moves the global hand
        # alignment block selected by palm_joint_indices().
        frozen = np.ones(n, dtype=bool)
        frozen[block_joints] = False

        step_limits = np.full(n, args.max_step, dtype=np.float64)
        for idx in block_joints:
            name = hand.pk_chain.get_joint_parameter_names()[idx]
            if name in {"virtual_joint_x", "virtual_joint_y", "virtual_joint_z"}:
                step_limits[idx] = args.palm_translation_step
            elif name.startswith("virtual_joint_"):
                step_limits[idx] = args.palm_rotation_step
            else:
                step_limits[idx] = args.palm_wrist_step

        objective_terms = [
            args.step_weight * cp.sum_squares(dq),
            args.palm_anchor_weight
            * cp.sum_squares(q_np[block_joints] + dq[block_joints] - q_ref_np[block_joints]),
        ]
        constraints = [
            dq[frozen] == 0.0,
            dq[block_joints] <= np.minimum(step_limits[block_joints], upper[block_joints] - q_np[block_joints]),
            dq[block_joints] >= np.maximum(-step_limits[block_joints], lower[block_joints] - q_np[block_joints]),
        ]

        for finger_idx, enabled in enumerate(finger_mask):
            if not enabled:
                continue
            add_surface_collision_constraints(
                hand,
                q_current,
                status,
                dq,
                objective_terms,
                constraints,
                finger_idx,
                object_pc_normals,
                args,
            )
            add_contact_band_constraints(
                hand,
                q_current,
                status,
                dq,
                objective_terms,
                constraints,
                finger_idx,
                object_pc_normals,
                args,
            )

        if args.palm_reach_objective:
            object_pc = object_pc_normals[:, :3].to(device)
            normals = object_pc_normals[:, 3:].to(device)
            active_fingers = [idx for idx, enabled in enumerate(finger_mask) if enabled]
            for finger_idx in active_fingers:
                links = [
                    name
                    for name in allowed_target_links(finger_idx, args.palm_reach_link_scope)
                    if name in status and name in hand.links_pc
                ]
                if not links:
                    continue
                link_jacs = jacobian(hand.pk_chain, q_current, status, links)
                for link_name in links:
                    local_pc = sample_points(hand.links_pc[link_name].to(device), args.palm_reach_points)
                    local_pc = subset_for_candidate(
                        link_name,
                        local_pc,
                        "distal_center" if link_name.endswith("distal") else (
                            "middle_center" if link_name.endswith("middle") else "proximal_center"
                        ),
                        args,
                    )
                    if local_pc.numel() == 0:
                        continue
                    se3 = status[link_name].get_matrix()[0].to(device)
                    ones = torch.ones(local_pc.shape[0], 1, dtype=local_pc.dtype, device=device)
                    world_pc = (torch.cat([local_pc, ones], dim=1) @ se3.T)[:, :3]
                    signed, unsigned = signed_distances(world_pc, object_pc, normals)
                    keep = min(args.palm_reach_k, world_pc.shape[0])
                    candidate_idx = torch.topk(unsigned, keep, largest=False).indices
                    link_jac = link_jacs[link_name][0].detach().cpu().numpy().astype(np.float64)
                    for point_idx in candidate_idx.detach().cpu().tolist():
                        point_xyz_t = world_pc[point_idx]
                        distances = torch.cdist(point_xyz_t[None, None, :], object_pc[None, :, :])[0, 0]
                        nearest_idx = int(torch.argmin(distances).detach().cpu())
                        target_t = object_pc[nearest_idx] + args.palm_reach_target_distance * normals[nearest_idx]
                        point_xyz = point_xyz_t.detach().cpu().numpy().astype(np.float64)
                        target_xyz = target_t.detach().cpu().numpy().astype(np.float64)
                        point_jac = point_jacobian_from_link(se3, link_jac, point_xyz)
                        objective_terms.append(
                            args.palm_reach_weight * cp.sum_squares(point_xyz + point_jac @ dq - target_xyz)
                        )

        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            problem.solve(
                solver="OSQP",
                warm_start=True,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=args.osqp_max_iter,
            )
        except Exception:
            problem.solve(solver="CLARABEL", verbose=False)
        last_status = problem.status
        last_value = float(problem.value) if problem.value is not None else None
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT} or dq.value is None:
            break
        delta = torch.as_tensor(dq.value, dtype=q_current.dtype, device=device)
        q_current = q_current + args.palm_line_search * delta
        if float(delta[block_joints].norm().detach().cpu()) < args.converge_delta:
            break

    return q_current.detach(), {
        "phase": "palm",
        "status": last_status,
        "last_objective": last_value,
        "joint_names": [hand.pk_chain.get_joint_parameter_names()[idx] for idx in block_joints],
    }

def sequential_qp_refine(
    hand,
    q_start: torch.Tensor,
    q_full: torch.Tensor,
    disabled_responsibility: np.ndarray,
    finger_mask: list[int],
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    object_pc_normals: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    """Run the complete active-finger -> thumb -> palm sequence.

    Pseudocode:
        q = sparse start from q_full
        for cycle:
            for active non-thumb finger f:
                recompute R_res(q)
                solve distal/middle/proximal hypotheses
                accept or reject best candidate
            solve thumb support QP
            solve palm/wrist prealignment QP
    """

    q = q_start.detach().clone()
    if float(np.sum(disabled_responsibility)) <= 1e-12:
        return q, {
            "finger_order": [FINGER_NAMES[idx] for idx in ordered_active_nonthumb(finger_mask)],
            "history": [],
            "final_residual_mass": 0.0,
        }
    order = ordered_active_nonthumb(finger_mask)
    history = []
    patch_kernel = build_patch_compensation_kernel(
        patch_positions,
        patch_normals,
        sigma_pos=args.patch_kernel_sigma_pos,
        sigma_normal=args.patch_kernel_sigma_normal,
        sigma_wrench=args.patch_kernel_sigma_wrench,
        wrench_weight=args.patch_kernel_wrench_weight,
    )

    def finalize_candidate(
        q_before_step: torch.Tensor,
        q_candidate: torch.Tensor,
        step_info: dict,
        before_residual: float,
        before_stats: dict,
        moving_finger_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Apply phase-specific acceptance and attach common diagnostics."""

        after_residual = reduced_residual_mass(
            hand,
            q_candidate,
            disabled_responsibility,
            finger_mask,
            patch_positions,
            patch_normals,
            args,
        )
        after_global_stats = reduced_penetration_stats(
            hand,
            q_candidate,
            object_pc_normals,
            finger_mask,
            args,
        )
        before_accept_stats = before_stats
        after_accept_stats = after_global_stats
        penetration_scope = "active_hand"
        before_finger_gap = None
        after_finger_gap = None
        if moving_finger_idx is not None:
            moving_mask = [1 if idx == moving_finger_idx else 0 for idx in range(len(FINGER_NAMES))]
            before_accept_stats = reduced_penetration_stats(
                hand,
                q_before_step,
                object_pc_normals,
                moving_mask,
                args,
            )
            after_accept_stats = reduced_penetration_stats(
                hand,
                q_candidate,
                object_pc_normals,
                moving_mask,
                args,
            )
            penetration_scope = FINGER_NAMES[moving_finger_idx]
            before_finger_gap = single_finger_gap_stats(
                hand,
                q_before_step,
                object_pc_normals,
                moving_finger_idx,
                args,
            )
            after_finger_gap = single_finger_gap_stats(
                hand,
                q_candidate,
                object_pc_normals,
                moving_finger_idx,
                args,
            )
        phase_name = step_info.get("phase")
        before_wrench = wrench_balance_stats(
            hand,
            q_before_step,
            finger_mask,
            patch_positions,
            patch_normals,
            args,
            include_thumb=True,
        )
        after_wrench = wrench_balance_stats(
            hand,
            q_candidate,
            finger_mask,
            patch_positions,
            patch_normals,
            args,
            include_thumb=True,
        )
        before_opposition = None
        after_opposition = None
        if phase_name == "active_finger" and args.active_objective_mode == "responsibility_gap":
            full_c, _, _, _ = compute_finger_responsibility(hand, q_full, patch_positions, patch_normals, args)
            target = target_responsibility(
                full_c,
                disabled_responsibility,
                finger_mask,
                disabled_weight=args.disabled_compensation_weight,
            )
            rows = active_nonthumb_rows(finger_mask)
            before_c, _, _, _ = compute_finger_responsibility(hand, q_before_step, patch_positions, patch_normals, args)
            after_c, _, _, _ = compute_finger_responsibility(hand, q_candidate, patch_positions, patch_normals, args)
            before_active = before_c[rows].sum(axis=0) if rows else np.zeros(patch_positions.shape[0])
            after_active = after_c[rows].sum(axis=0) if rows else np.zeros(patch_positions.shape[0])
            before_gap_kernel = float(positive_gap(target, before_active, patch_kernel).sum())
            after_gap_kernel = float(positive_gap(target, after_active, patch_kernel).sum())
            before_self = step_info.get("self_retention_before")
            after_self = step_info.get("self_retention_after")
            self_floor = min(args.responsibility_min_self_retention, before_self if before_self is not None else 1.0)
            self_ok = after_self is None or after_self >= self_floor - 0.05
            penetration_guard_ok = accept_penetration_guard(
                before_accept_stats,
                after_accept_stats,
                args.accept_max_penetration_worsen_mm,
                args,
            )
            accepted = (
                after_gap_kernel <= before_gap_kernel + args.accept_residual_worsen
                and self_ok
                and penetration_guard_ok
            )
            step_info.update(
                {
                    "gap_kernel_before": before_gap_kernel,
                    "gap_kernel_after": after_gap_kernel,
                    "gap_kernel_gain": before_gap_kernel - after_gap_kernel,
                    "self_retention_guard_ok": self_ok,
                    "penetration_guard_ok": penetration_guard_ok,
                }
            )
        elif phase_name == "thumb":
            before_opposition = thumb_opposition_stats(
                hand,
                q_before_step,
                finger_mask,
                object_pc_normals,
                patch_positions,
                args,
            )
            after_opposition = thumb_opposition_stats(
                hand,
                q_candidate,
                finger_mask,
                object_pc_normals,
                patch_positions,
                args,
            )
            accepted = accept_thumb_step(
                before_accept_stats,
                after_accept_stats,
                before_finger_gap or {},
                after_finger_gap or {},
                before_opposition,
                after_opposition,
                before_wrench,
                after_wrench,
                args,
            )
        elif phase_name == "palm":
            accepted = accept_palm_step(
                before_accept_stats,
                after_accept_stats,
                before_wrench,
                after_wrench,
                args,
            )
        else:
            accepted = accept_candidate_step(
                before_residual,
                after_residual,
                before_accept_stats,
                after_accept_stats,
                args,
                before_finger_gap,
                after_finger_gap,
            )
        if accepted:
            q_next = q_candidate
            after_stats = after_global_stats
        else:
            q_next = q_before_step
            candidate_residual = after_residual
            candidate_global_stats = after_global_stats
            candidate_accept_stats = after_accept_stats
            candidate_finger_gap = after_finger_gap
            candidate_wrench = after_wrench
            candidate_opposition = after_opposition
            after_residual = before_residual
            after_stats = before_stats
            after_accept_stats = before_accept_stats
            after_finger_gap = before_finger_gap
            after_wrench = before_wrench
            after_opposition = before_opposition
            step_info["status"] = f"{step_info.get('status')}_rejected"
            step_info["candidate_residual_after"] = candidate_residual
            step_info["candidate_global_max_penetration_after_mm"] = candidate_global_stats["max_penetration_mm"]
            step_info["candidate_global_near_after"] = candidate_global_stats["near_ratio"]
            step_info["candidate_acceptance_max_penetration_after_mm"] = candidate_accept_stats["max_penetration_mm"]
            step_info["candidate_acceptance_near_after"] = candidate_accept_stats["near_ratio"]
            step_info["candidate_moving_finger_gap_after"] = candidate_finger_gap
            step_info["candidate_wrench_after"] = candidate_wrench
            step_info["candidate_thumb_opposition_after"] = candidate_opposition
        responsibility_before = responsibility_consistency_stats(
            before_residual,
            disabled_responsibility,
        )
        responsibility_after = responsibility_consistency_stats(
            after_residual,
            disabled_responsibility,
        )
        step_info.update(
            {
                "cycle": cycle,
                "residual_before": before_residual,
                "residual_after": after_residual,
                "responsibility_before": responsibility_before,
                "responsibility_after": responsibility_after,
                "responsibility_coverage_before": responsibility_before["coverage_ratio"],
                "responsibility_coverage_after": responsibility_after["coverage_ratio"],
                "accepted": accepted,
                "penetration_scope": penetration_scope,
                "near_before": before_accept_stats["near_ratio"],
                "near_after": after_accept_stats["near_ratio"],
                "max_penetration_before_mm": before_accept_stats["max_penetration_mm"],
                "max_penetration_after_mm": after_accept_stats["max_penetration_mm"],
                "global_near_before": before_stats["near_ratio"],
                "global_near_after": after_stats["near_ratio"],
                "global_max_penetration_before_mm": before_stats["max_penetration_mm"],
                "global_max_penetration_after_mm": after_stats["max_penetration_mm"],
                "moving_finger": FINGER_NAMES[moving_finger_idx] if moving_finger_idx is not None else None,
                "moving_finger_gap_before": before_finger_gap,
                "moving_finger_gap_after": after_finger_gap,
                "wrench_before": before_wrench,
                "wrench_after": after_wrench,
                "thumb_opposition_before": before_opposition,
                "thumb_opposition_after": after_opposition,
            }
        )
        return q_next, step_info

    def solve_active_finger_step(
        finger_idx: int,
        residual: np.ndarray,
        before_residual: float,
        q_before_step: torch.Tensor,
        before_stats: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Try one or more contact-part hypotheses for an active finger."""

        if args.active_objective_mode == "responsibility_gap":
            return solve_single_finger_responsibility_gap_qp(
                hand,
                q_before_step,
                q_full,
                q_full,
                finger_idx,
                finger_mask,
                disabled_responsibility,
                patch_positions,
                patch_normals,
                patch_kernel,
                object_pc_normals,
                args,
            )

        if args.part_assignment_mode != "multi_hypothesis":
            targets = select_targets_for_finger(
                hand,
                q,
                finger_idx,
                residual,
                patch_positions,
                patch_normals,
                args,
            )
            q_candidate, step_info = solve_single_finger_qp(
                hand,
                q,
                q_full,
                finger_idx,
                finger_mask,
                targets,
                object_pc_normals,
                args,
            )
            step_info["hypothesis_part"] = "auto"
            step_info["hypothesis_score"] = None
            step_info["hypothesis_alternatives"] = []
            return q_candidate, step_info

        before_finger_gap = single_finger_gap_stats(
            hand,
            q_before_step,
            object_pc_normals,
            finger_idx,
            args,
        )
        finger_only_mask = [1 if idx == finger_idx else 0 for idx in range(len(FINGER_NAMES))]
        before_finger_stats = reduced_penetration_stats(
            hand,
            q_before_step,
            object_pc_normals,
            finger_only_mask,
            args,
        )
        before_wrench = wrench_balance_stats(
            hand,
            q_before_step,
            finger_mask,
            patch_positions,
            patch_normals,
            args,
            include_thumb=True,
        )
        hypotheses = []
        for contact_part in args.contact_part_candidates:
            targets = select_targets_for_finger(
                hand,
                q,
                finger_idx,
                residual,
                patch_positions,
                patch_normals,
                args,
                contact_part=contact_part,
            )
            if not targets:
                continue
            q_candidate, step_info = solve_single_finger_qp(
                hand,
                q,
                q_full,
                finger_idx,
                finger_mask,
                targets,
                object_pc_normals,
                args,
            )
            after_residual = reduced_residual_mass(
                hand,
                q_candidate,
                disabled_responsibility,
                finger_mask,
                patch_positions,
                patch_normals,
                args,
            )
            after_global_stats = reduced_penetration_stats(
                hand,
                q_candidate,
                object_pc_normals,
                finger_mask,
                args,
            )
            after_finger_stats = reduced_penetration_stats(
                hand,
                q_candidate,
                object_pc_normals,
                finger_only_mask,
                args,
            )
            after_finger_gap = single_finger_gap_stats(
                hand,
                q_candidate,
                object_pc_normals,
                finger_idx,
                args,
            )
            after_wrench = wrench_balance_stats(
                hand,
                q_candidate,
                finger_mask,
                patch_positions,
                patch_normals,
                args,
                include_thumb=True,
            )
            would_accept = accept_candidate_step(
                before_residual,
                after_residual,
                before_finger_stats,
                after_finger_stats,
                args,
                before_finger_gap,
                after_finger_gap,
            )
            score = hypothesis_score(
                hand,
                q_before_step,
                q_candidate,
                finger_idx,
                before_residual,
                after_residual,
                before_finger_stats,
                after_finger_stats,
                before_finger_gap,
                after_finger_gap,
                before_wrench,
                after_wrench,
                would_accept,
                args,
            )
            step_info.update(
                {
                    "hypothesis_part": contact_part,
                    "hypothesis_score": score,
                    "hypothesis_would_accept": would_accept,
                    "hypothesis_residual_after": after_residual,
                    "hypothesis_penetration_scope": FINGER_NAMES[finger_idx],
                    "hypothesis_near_after": after_finger_stats["near_ratio"],
                    "hypothesis_max_penetration_after_mm": after_finger_stats["max_penetration_mm"],
                    "hypothesis_global_near_after": after_global_stats["near_ratio"],
                    "hypothesis_global_max_penetration_after_mm": after_global_stats["max_penetration_mm"],
                    "hypothesis_moving_finger_gap_after": after_finger_gap,
                    "hypothesis_wrench_before": before_wrench,
                    "hypothesis_wrench_after": after_wrench,
                }
            )
            hypotheses.append((score, q_candidate, step_info))

        if not hypotheses:
            return q_before_step, {
                "finger": FINGER_NAMES[finger_idx],
                "num_targets": 0,
                "status": "no_part_targets",
                "hypothesis_part": None,
                "hypothesis_score": None,
                "hypothesis_alternatives": [],
            }

        hypotheses.sort(key=lambda item: item[0], reverse=True)
        best_score, best_q, best_info = hypotheses[0]
        alternatives = []
        for score, _, info in hypotheses:
            alternatives.append(
                {
                    "part": info.get("hypothesis_part"),
                    "score": score,
                    "would_accept": info.get("hypothesis_would_accept"),
                    "status": info.get("status"),
                    "target_links": info.get("target_links", []),
                    "target_candidates": info.get("target_candidates", []),
                    "residual_after": info.get("hypothesis_residual_after"),
                    "near_after": info.get("hypothesis_near_after"),
                    "max_penetration_after_mm": info.get("hypothesis_max_penetration_after_mm"),
                    "moving_finger_gap_after": info.get("hypothesis_moving_finger_gap_after"),
                    "wrench_after": info.get("hypothesis_wrench_after"),
                }
            )
        best_info["hypothesis_score"] = best_score
        best_info["hypothesis_alternatives"] = alternatives
        return best_q, best_info

    for cycle in range(args.cycles):
        for finger_idx in order:
            # Responsibility is recomputed before every finger move, so targets
            # follow the residual that remains after earlier accepted steps.
            current_c, _, _, _ = compute_finger_responsibility(
                hand,
                q,
                patch_positions,
                patch_normals,
                args,
            )
            active_rows = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
            current_total = current_c[active_rows].sum(axis=0)
            residual = np.maximum(0.0, disabled_responsibility - current_total)
            targets = select_targets_for_finger(
                hand,
                q,
                finger_idx,
                residual,
                patch_positions,
                patch_normals,
                args,
            )
            before_residual = float(residual.sum())
            q_before_step = q.detach().clone()
            before_stats = reduced_penetration_stats(
                hand,
                q_before_step,
                object_pc_normals,
                finger_mask,
                args,
            )
            q_candidate, step_info = solve_active_finger_step(
                finger_idx,
                residual,
                before_residual,
                q_before_step,
                before_stats,
            )
            step_info["phase"] = "active_finger"
            q, step_info = finalize_candidate(
                q_before_step,
                q_candidate,
                step_info,
                before_residual,
                before_stats,
                moving_finger_idx=finger_idx,
            )
            history.append(step_info)

        if args.thumb_phase and finger_mask[0]:
            before_residual = reduced_residual_mass(
                hand,
                q,
                disabled_responsibility,
                finger_mask,
                patch_positions,
                patch_normals,
                args,
            )
            q_before_step = q.detach().clone()
            before_stats = reduced_penetration_stats(
                hand,
                q_before_step,
                object_pc_normals,
                finger_mask,
                args,
            )
            thumb_targets = select_thumb_targets(
                hand,
                q,
                finger_mask,
                patch_positions,
                patch_normals,
                object_pc_normals,
                args,
            )
            q_candidate, step_info = solve_single_finger_qp(
                hand,
                q,
                q_full,
                0,
                finger_mask,
                thumb_targets,
                object_pc_normals,
                args,
            )
            step_info["phase"] = "thumb"
            q, step_info = finalize_candidate(
                q_before_step,
                q_candidate,
                step_info,
                before_residual,
                before_stats,
                moving_finger_idx=0,
            )
            history.append(step_info)

        if args.palm_phase:
            before_residual = reduced_residual_mass(
                hand,
                q,
                disabled_responsibility,
                finger_mask,
                patch_positions,
                patch_normals,
                args,
            )
            q_before_step = q.detach().clone()
            before_stats = reduced_penetration_stats(
                hand,
                q_before_step,
                object_pc_normals,
                finger_mask,
                args,
            )
            q_candidate, step_info = solve_palm_qp(
                hand,
                q,
                q_full,
                finger_mask,
                object_pc_normals,
                args,
            )
            q, step_info = finalize_candidate(
                q_before_step,
                q_candidate,
                step_info,
                before_residual,
                before_stats,
            )
            history.append(step_info)

    final_c, _, _, _ = compute_finger_responsibility(hand, q, patch_positions, patch_normals, args)
    active_rows = [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]
    final_residual = np.maximum(0.0, disabled_responsibility - final_c[active_rows].sum(axis=0))
    return q, {
        "finger_order": [FINGER_NAMES[idx] for idx in order],
        "phase_order": ["active_fingers", "thumb", "palm"],
        "history": history,
        "final_residual_mass": float(final_residual.sum()),
    }

@dataclass
class SequentialQP:
    """CFET Part B: sequential QP workflow."""

    config: CFETConfig

    def __post_init__(self) -> None:
        self.robot = RobotProcessor(self.config)
        self.object = ObjectProcessor(self.config)
        self.args = self.robot.args
        self.object_pc_normals = self.object.load_point_cloud_with_normals(self.robot.device)
        self.patch_positions, self.patch_normals = self.object.make_patches(self.object_pc_normals)
        self.q_full = self.robot.load_full_grasp()
        self.responsibility = ResponsibilityComputer(
            self.robot.hand,
            self.patch_positions,
            self.patch_normals,
            self.args,
        )

    def inspect_responsibility(self, mode_name: str, finger_idx: int) -> dict[str, Any]:
        root = output_root(self.config.output_dir)
        self.responsibility.write_link_patch_csv(root / "A0_link_patch.csv", self.q_full)
        finger_mask = self.robot.finger_mask(mode_name)
        q_start = self.robot.sparse_start_from_mode(self.q_full, mode_name)
        disabled = self.responsibility.disabled_responsibility(self.q_full, finger_mask)
        residual = self.responsibility.residual(q_start, disabled, finger_mask)
        targets = self.responsibility.select_targets(q_start, finger_idx, residual)
        return {
            "object": self.config.object_name,
            "mode": mode_name,
            "disabled_mass": float(disabled.sum()),
            "residual_mass": float(residual.sum()),
            "target_patches": [int(target.patch_idx) for target in targets],
            "target_weights": [float(target.weight) for target in targets],
            "link_patch_csv": str(root / "A0_link_patch.csv"),
        }

    def run_mode(self, mode_name: str) -> dict[str, Any]:
        root = output_root(self.config.output_dir)
        mode_dir = root / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        finger_mask = self.robot.finger_mask(mode_name)
        q_start = self.robot.sparse_start_from_mode(self.q_full, mode_name)
        disabled = self.responsibility.disabled_responsibility(self.q_full, finger_mask)
        q_refined, sequence = sequential_qp_refine(
            self.robot.hand,
            q_start,
            self.q_full,
            disabled,
            finger_mask,
            self.patch_positions,
            self.patch_normals,
            self.object_pc_normals,
            self.args,
        )

        c_start, _, _, _ = self.responsibility.compute(q_start)
        c_final, _, _, _ = self.responsibility.compute(q_refined)
        write_cstar_csv(mode_dir / "responsibility.csv", c_start, c_final)
        torch.save({"q_start": q_start.detach().cpu(), "q_refined": q_refined.detach().cpu()}, mode_dir / "q_result.pt")

        start_residual = np.maximum(
            0.0,
            disabled - self.responsibility.active_coverage(q_start, finger_mask),
        )
        final_residual = np.maximum(
            0.0,
            disabled - self.responsibility.active_coverage(q_refined, finger_mask),
        )

        still_path = None
        if self.config.render:
            renderer = MujocoRenderer(str(root), title=f"F2M {self.config.object_name}")
            renderer.export_scene_preview(self.robot.hand, q_start, self.object_pc_normals, mode_dir, name="before")
            still_path = str(renderer.export_scene_preview(self.robot.hand, q_refined, self.object_pc_normals, mode_dir, name="after"))

        stats = {
            "object": self.config.object_name,
            "mode": mode_name,
            "disabled_mass": float(disabled.sum()),
            "start_residual_mass": float(start_residual.sum()),
            "final_residual_mass": float(final_residual.sum()),
            "before": reduced_penetration_stats(
                self.robot.hand, q_start, self.object_pc_normals, finger_mask, self.args
            ),
            "after": reduced_penetration_stats(
                self.robot.hand, q_refined, self.object_pc_normals, finger_mask, self.args
            ),
            "sequence": sequence,
            "still_path": still_path,
        }
        save_json(mode_dir / "stats.json", stats)
        if self.config.render:
            renderer.export_comparison_html(
                self.robot.hand,
                q_start,
                q_refined,
                self.object_pc_normals,
                mode_dir,
                stats=stats,
            )
            renderer.make_index()
        return stats

    def run_all_modes(self) -> list[dict[str, Any]]:
        results = [self.run_mode(mode) for mode in self.config.modes]
        root = output_root(self.config.output_dir)
        save_json(root / "render_stats.json", results)
        return results
