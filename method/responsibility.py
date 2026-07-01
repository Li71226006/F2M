from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contact_targets import compute_finger_responsibility, select_targets_for_finger
from .graph_utils import write_link_patch_csv as write_local_link_patch_csv


@dataclass
class ResponsibilityComputer:
    """Part A facade: responsibility weights and target selection.

    Use this class when other code needs the paper-level objects R_D, R_res,
    or C*_f without knowing the lower-level link/patch graph details.
    """

    hand: object
    patch_positions: np.ndarray
    patch_normals: np.ndarray
    args: object

    def __post_init__(self) -> None:
        self.write_link_patch = write_local_link_patch_csv

    def compute(self, q):
        """Return finger-patch and link-patch responsibility at pose q."""

        finger_patch, link_patch, link_names, link_fingers = compute_finger_responsibility(
            self.hand,
            q,
            self.patch_positions,
            self.patch_normals,
            self.args,
        )
        return finger_patch, link_patch, link_names, link_fingers

    def disabled_responsibility(self, q_full, finger_mask: list[int], *, include_thumb: bool = False) -> np.ndarray:
        """Compute R_D(k): responsibility mass of disabled fingers in q_full."""

        finger_patch, _, _, _ = self.compute(q_full)
        rows = [
            idx
            for idx, enabled in enumerate(finger_mask)
            if not enabled and (include_thumb or idx != 0)
        ]
        if not rows:
            return np.zeros(finger_patch.shape[1], dtype=np.float64)
        return finger_patch[rows].sum(axis=0)

    def active_coverage(self, q, finger_mask: list[int], *, include_thumb: bool = False) -> np.ndarray:
        """Compute R_active(k, q): patch coverage by enabled fingers."""

        finger_patch, _, _, _ = self.compute(q)
        rows = [
            idx
            for idx, enabled in enumerate(finger_mask)
            if enabled and (include_thumb or idx != 0)
        ]
        if not rows:
            return np.zeros(finger_patch.shape[1], dtype=np.float64)
        return finger_patch[rows].sum(axis=0)

    def residual(self, q, disabled_responsibility: np.ndarray, finger_mask: list[int]) -> np.ndarray:
        """Compute R_res(k, q) = max(0, R_D(k) - R_active(k, q))."""

        active = self.active_coverage(q, finger_mask, include_thumb=False)
        return np.maximum(0.0, disabled_responsibility - active)

    def select_targets(self, q, finger_idx: int, residual: np.ndarray):
        """Select C*_f targets for a specific finger from residual mass."""

        return select_targets_for_finger(
            self.hand,
            q,
            finger_idx,
            residual,
            self.patch_positions,
            self.patch_normals,
            self.args,
        )

    def write_link_patch_csv(self, path: Path, q) -> None:
        _, link_patch, link_names, link_fingers = self.compute(q)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_link_patch(path, link_patch, link_names, link_fingers)
