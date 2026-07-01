from __future__ import annotations

from dataclasses import dataclass

from .config import CFETConfig
from .constants import FINGER_JOINT_PREFIXES, FINGER_NAMES, MODES
from .func import build_solver_args, load_q, resolve_project_path
from .hand_model import create_hand_model


@dataclass
class RobotProcessor:
    """Robot/URDF/hand-state processing for the local F2M method."""

    config: CFETConfig

    def __post_init__(self) -> None:
        self.modes = MODES
        torch = __import__("torch")
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        if not self.config.urdf_path:
            raise ValueError("F2M needs config.urdf_path or --urdf for the local method.")
        self.args = build_solver_args(self.config)
        self.hand = create_hand_model(
            self.config.robot_name,
            resolve_project_path(self.config.urdf_path),
            self.device,
            resolve_project_path(self.config.robot_link_pc_path) if self.config.robot_link_pc_path else None,
        )

    def load_full_grasp(self):
        if self.config.initial_q_path:
            q = load_q(self.config.initial_q_path, self.device)
            if q.numel() != self.hand.dof:
                raise ValueError(f"Initial q has {q.numel()} values, but robot DOF is {self.hand.dof}")
            return q
        return self.hand.get_canonical_q()

    def sparse_start_from_mode(self, q_full, mode_name: str):
        finger_mask = self.modes[mode_name]
        if all(finger_mask):
            return q_full.clone()

        torch = __import__("torch")
        lower, _ = self.hand.pk_chain.get_joint_limits()
        lower = torch.tensor(lower, dtype=q_full.dtype, device=q_full.device)
        joint_orders = self.hand.get_joint_orders()
        q_sparse = q_full.clone()
        for finger_name, enabled in zip(FINGER_NAMES, finger_mask):
            if enabled:
                continue
            prefix = FINGER_JOINT_PREFIXES[finger_name]
            for joint_idx, joint_name in enumerate(joint_orders):
                if joint_name.startswith(prefix):
                    q_sparse[joint_idx] = lower[joint_idx]
        return q_sparse

    def finger_mask(self, mode_name: str) -> list[int]:
        return self.modes[mode_name]
