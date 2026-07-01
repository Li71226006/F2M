from __future__ import annotations

"""Responsibility-gap objective utilities.

This module supports the v2 objective:

    minimize positive_part(T - K^T R_active(q))

where T contains the active fingers' original full-hand responsibility plus
the disabled-finger responsibility to be compensated. K is a patch compensation
kernel that lets nearby, normal-compatible, wrench-compatible patches partially
substitute for one another.
"""

import numpy as np
import torch

from .contact_targets import compute_finger_responsibility, finger_joint_indices, patch_wrench_matrix


def build_patch_compensation_kernel(
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    *,
    sigma_pos: float,
    sigma_normal: float,
    sigma_wrench: float,
    wrench_weight: float,
) -> np.ndarray:
    """Return K[a, b]: how much source patch a can compensate target patch b."""

    diff = patch_positions[:, None, :] - patch_positions[None, :, :]
    pos_cost = np.sum(diff * diff, axis=-1) / max(sigma_pos**2, 1e-12)

    normals = patch_normals / (np.linalg.norm(patch_normals, axis=1, keepdims=True) + 1e-12)
    normal_dot = np.clip(normals @ normals.T, -1.0, 1.0)
    normal_cost = (1.0 - normal_dot) / max(sigma_normal, 1e-12)

    wrenches = patch_wrench_matrix(patch_positions, normals)
    wrench_diff = wrenches[:, None, :] - wrenches[None, :, :]
    wrench_cost = np.sum(wrench_diff * wrench_diff, axis=-1) / max(sigma_wrench**2, 1e-12)

    kernel = np.exp(-(pos_cost + normal_cost + wrench_weight * wrench_cost))
    np.fill_diagonal(kernel, 1.0)
    return kernel.astype(np.float64)


def compensated_coverage(coverage: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Diffuse source patch responsibility through the compensation kernel."""

    return kernel.T @ coverage


def positive_gap(target: np.ndarray, coverage: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Positive residual after allowing patch-to-patch compensation."""

    return np.maximum(0.0, target - compensated_coverage(coverage, kernel))


def active_nonthumb_rows(finger_mask: list[int]) -> list[int]:
    return [idx for idx, enabled in enumerate(finger_mask) if enabled and idx != 0]


def target_responsibility(
    full_finger_patch: np.ndarray,
    disabled_responsibility: np.ndarray,
    finger_mask: list[int],
    *,
    disabled_weight: float,
) -> np.ndarray:
    """Target distribution: keep active responsibility and add disabled demand."""

    active_rows = active_nonthumb_rows(finger_mask)
    active_full = full_finger_patch[active_rows].sum(axis=0) if active_rows else np.zeros_like(disabled_responsibility)
    return active_full + disabled_weight * disabled_responsibility


def finite_difference_finger_responsibility_jacobian(
    hand,
    q: torch.Tensor,
    finger_idx: int,
    patch_positions: np.ndarray,
    patch_normals: np.ndarray,
    args,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference dA_f/dq_f for one active finger.

    Returns:
        base_finger: [K] current responsibility of this finger.
        jac_finger: [K, m] derivative wrt this finger's m joints.
        joints: global joint indices for those m joints.
    """

    joints = finger_joint_indices(hand, finger_idx)
    base, _, _, _ = compute_finger_responsibility(hand, q, patch_positions, patch_normals, args)
    base_finger = base[finger_idx].astype(np.float64)
    jac = np.zeros((patch_positions.shape[0], len(joints)), dtype=np.float64)
    lower, upper = hand.pk_chain.get_joint_limits()
    lower_t = torch.as_tensor(lower, dtype=q.dtype, device=q.device)
    upper_t = torch.as_tensor(upper, dtype=q.dtype, device=q.device)

    for local_col, joint_idx in enumerate(joints):
        q_pos = q.detach().clone()
        q_neg = q.detach().clone()
        q_pos[joint_idx] = torch.minimum(q_pos[joint_idx] + eps, upper_t[joint_idx])
        q_neg[joint_idx] = torch.maximum(q_neg[joint_idx] - eps, lower_t[joint_idx])
        denom = float((q_pos[joint_idx] - q_neg[joint_idx]).detach().cpu())
        if abs(denom) < 1e-12:
            continue
        c_pos, _, _, _ = compute_finger_responsibility(hand, q_pos, patch_positions, patch_normals, args)
        c_neg, _, _, _ = compute_finger_responsibility(hand, q_neg, patch_positions, patch_normals, args)
        jac[:, local_col] = (c_pos[finger_idx] - c_neg[finger_idx]) / denom
    return base_finger, jac, joints


def self_retention_ratio(current_finger: np.ndarray, full_finger: np.ndarray, kernel: np.ndarray) -> float:
    """How much of this finger's original full-hand responsibility remains."""

    target_mass = float(np.sum(full_finger))
    if target_mass <= 1e-12:
        return 1.0
    current_comp = compensated_coverage(current_finger, kernel)
    retained = np.minimum(current_comp, full_finger).sum()
    return float(retained / (target_mass + 1e-12))


def raw_self_retention_ratio(current_finger: np.ndarray, full_finger: np.ndarray) -> float:
    """Patch-local retention without compensation-kernel diffusion.

    This stricter metric prevents a finger from losing its own patch-level role
    while appearing acceptable only because nearby patches diffuse through K.
    """

    target_mass = float(np.sum(full_finger))
    if target_mass <= 1e-12:
        return 1.0
    retained = np.minimum(current_finger, full_finger).sum()
    return float(retained / (target_mass + 1e-12))
