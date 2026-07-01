from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CFETConfig


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return workspace_root() / "F2M"


def output_root(output_dir: str) -> Path:
    root = Path(output_dir)
    if not root.is_absolute():
        root = project_root() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [project_root() / path, workspace_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def signed_distance_proxy(points: np.ndarray, surface_points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Point-cloud signed-distance proxy.

    This is not a watertight mesh SDF. It projects each query point onto the
    nearest surface sample normal, matching the current TRO prototype.
    """

    diff = points[:, None, :] - surface_points[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    nearest = np.argmin(d2, axis=1)
    nearest_vec = points - surface_points[nearest]
    return np.sum(nearest_vec * normals[nearest], axis=-1)


def load_q(path: str | Path, device: torch.device) -> torch.Tensor:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Initial q file not found: {path}")
    if path.suffix.lower() == ".pt":
        value = torch.load(path, map_location=device)
        if isinstance(value, dict):
            for key in ("q", "target_q", "initial_q", "q_full"):
                if key in value:
                    value = value[key]
                    break
    elif path.suffix.lower() in {".npy", ".npz"}:
        value = np.load(path)
        if isinstance(value, np.lib.npyio.NpzFile):
            key = "q" if "q" in value else value.files[0]
            value = value[key]
    else:
        value = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    return torch.as_tensor(value, dtype=torch.float32, device=device).flatten()


def build_solver_args(config: CFETConfig) -> Namespace:
    """Collect internal solver knobs used by responsibility and QP modules."""

    values: dict[str, Any] = {
        "object": config.object_name,
        "robot": config.robot_name,
        "modes": config.modes,
        "device": config.device,
        "output_dir": config.output_dir,
        "num_patches": config.num_patches,
        "c0_top_k": config.c0_top_k,
        "c0_sigma": config.c0_sigma,
        "link_radius": config.link_radius,
        "responsibility_signed_band": True,
        "responsibility_outside_band": 0.012,
        "responsibility_tangent_sigma": 0.030,
        "responsibility_penetration_sigma": 0.012,
        "link_points_per_link": 96,
        "no_direction_factor": False,
        "penetration_saturates_affinity": True,
        "affinity_contact_distance": 0.0015,
        "affinity_contact_radius": 0.020,
        "target_link_scope": "distal_middle",
        "target_surface_points": 96,
        "target_contact_candidate_mode": "candidate_pad",
        "target_assignment_mode": config.target_assignment_mode,
        "residual_regions_per_finger": config.residual_regions_per_finger,
        "residual_region_radius": config.residual_region_radius,
        "candidate_pool_allow_multi_link_region": config.candidate_pool_allow_multi_link_region,
        "target_candidate_fraction": 0.35,
        "target_candidate_priority_weight": 0.010,
        "target_normal_weight": 0.010,
        "target_link_priority_mode": "distal_first",
        "target_distal_max_distance": 0.070,
        "target_middle_fallback": True,
        "part_assignment_mode": "multi_hypothesis",
        "contact_part_candidates": ["distal", "middle", "proximal"],
        "part_prior_distal": 1.00,
        "part_prior_middle": 0.85,
        "part_prior_proximal": 0.35,
        "hypothesis_residual_weight": 1.0,
        "hypothesis_near_weight": 12.0,
        "hypothesis_finger_distance_weight": 0.18,
        "hypothesis_finger_near_weight": 8.0,
        "hypothesis_wrench_weight": 0.8,
        "hypothesis_penetration_weight": 0.15,
        "hypothesis_joint_delta_weight": 0.01,
        "hypothesis_reject_penalty": 2.0,
        "target_point_mode": "hard_nearest",
        "target_softmin_temp": 1e-4,
        "targets_per_finger": 4,
        "target_gap": 0.003,
        "target_reach_sigma": 0.070,
        "min_target_weight": 0.02,
        "nms_radius": 0.018,
        "cycles": config.cycles,
        "qp_iters": config.qp_iters,
        "osqp_max_iter": 30000,
        "max_step": 0.10,
        "line_search": 0.75,
        "converge_delta": 1e-3,
        "contact_weight": 260.0,
        "target_direction_weight": 45.0,
        "target_progress_fraction": 0.35,
        "target_max_progress": 0.012,
        "joint_anchor_weight": 0.4,
        "step_weight": 0.02,
        "surface_collision": True,
        "surface_collision_points": 48,
        "surface_collision_k": 12,
        "surface_collision_margin": 0.012,
        "allowed_penetration": config.allowed_penetration,
        "collision_slack_weight": 80000.0,
        "contact_band": config.contact_band,
        "contact_band_link_scope": "distal_middle",
        "contact_band_points": 64,
        "contact_band_k": 4,
        "contact_band_max_distance": 0.012,
        "contact_band_weight": 9000.0,
        "contact_band_penetration": True,
        "contact_band_candidate_mode": "exterior",
        "self_collision": True,
        "self_collision_include_thumb": True,
        "self_collision_palm_links": ["palm"],
        "self_collision_points": 32,
        "self_collision_k": 12,
        "self_collision_margin": 0.018,
        "self_collision_min_distance": 0.010,
        "self_collision_slack_weight": 12000.0,
        "acceptance_guard": True,
        "accept_residual_gain": 1e-4,
        "accept_near_gain": 0.01,
        "accept_near_drop": 0.01,
        "accept_penetration_gain_mm": 1.0,
        "accept_max_penetration_worsen_mm": 0.5,
        "accept_max_total_penetration_mm": 10.0,
        "accept_penetration_over_limit_worsen_mm": 0.1,
        "moving_finger_acceptance": True,
        "accept_finger_distance_gain_mm": 0.25,
        "accept_finger_near_gain": 0.002,
        "accept_finger_max_distance_mm": 14.0,
        "accept_finger_max_distance_worsen_mm": 0.75,
        "accept_contact_near_drop": 0.025,
        "accept_residual_worsen": 0.05,
        "thumb_phase": True,
        "thumb_contact_targets": True,
        "thumb_targets": 2,
        "thumb_target_link_scope": "distal_middle",
        "thumb_target_weight": 1.5,
        "thumb_reach_sigma": 0.090,
        "thumb_opposition_weight": 0.75,
        "thumb_normal_weight": 0.25,
        "thumb_wrench_weight": 0.35,
        "thumb_opposition_near_distance": 0.018,
        "thumb_opposition_fallback_points": 24,
        "thumb_accept_distance_gain_mm": 0.20,
        "thumb_accept_near_gain": 0.002,
        "thumb_accept_max_distance_mm": 8.0,
        "thumb_accept_max_distance_worsen_mm": 0.75,
        "thumb_accept_opposition_gain": 0.02,
        "thumb_accept_wrench_gain": 0.01,
        "thumb_accept_max_penetration_worsen_mm": 0.5,
        "palm_phase": True,
        "palm_phase_include_wrist": False,
        "palm_qp_iters": config.palm_qp_iters,
        "palm_translation_step": 0.006,
        "palm_rotation_step": 0.05,
        "palm_wrist_step": 0.04,
        "palm_line_search": 0.35,
        "palm_anchor_weight": 2.0,
        "palm_reach_objective": True,
        "palm_reach_link_scope": "distal_middle",
        "palm_reach_points": 32,
        "palm_reach_k": 2,
        "palm_reach_target_distance": 0.010,
        "palm_reach_weight": 90.0,
        "palm_accept_near_gain": 0.002,
        "palm_accept_wrench_gain": 0.01,
        "palm_accept_max_penetration_worsen_mm": 0.5,
        "active_objective_mode": config.active_objective_mode,
        "disabled_compensation_weight": 1.0,
        "patch_kernel_sigma_pos": 0.035,
        "patch_kernel_sigma_normal": 0.45,
        "patch_kernel_sigma_wrench": 1.0,
        "patch_kernel_wrench_weight": 0.15,
        "responsibility_fd_eps": 2e-3,
        "responsibility_gap_weight": 40.0,
        "responsibility_self_weight": 8.0,
        "responsibility_wrench_weight": 10.0,
        "responsibility_wrench_torque_scale": 8.0,
        "responsibility_direction_weight": 30.0,
        "responsibility_direction_targets": 2,
        "responsibility_min_self_retention": 0.65,
        "stats_points_per_link": 96,
        "near_distance": 0.012,
        "width": 900,
        "height": 650,
        "gif_frames": 24,
        "gif_fps": 12,
        "azimuth": 110.0,
        "elevation": -10.0,
        "distance": 0.23,
    }
    return Namespace(**values)


def render_args(args: Namespace, *, refine_active: bool) -> Namespace:
    return Namespace(
        width=args.width,
        height=args.height,
        gif_frames=args.gif_frames,
        gif_fps=args.gif_fps,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        refine_active=refine_active,
    )
