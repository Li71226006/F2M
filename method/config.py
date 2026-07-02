from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CFETConfig:
    """Public knobs for our method.

    Keep this small. Docker flags, MuJoCo details, and legacy TRO compatibility
    defaults should not grow here.
    """

    object_name: str = "object"
    robot_name: str = "shadowhand"
    urdf_path: str = ""
    robot_link_pc_path: str = ""
    object_point_cloud: str = ""
    initial_q_path: str = ""
    modes: list[str] = field(
        default_factory=lambda: ["5f_full", "4f_no_little", "3f_thumb_index_middle", "2f_thumb_index"]
    )
    device: str = "cpu"
    output_dir: str = "results/default"

    num_patches: int = 25
    c0_top_k: int = 5
    c0_sigma: float = 0.030
    link_radius: float = 0.080

    cycles: int = 1
    qp_iters: int = 4
    palm_qp_iters: int = 1
    contact_band: bool = False
    allowed_penetration: float = 0.003
    active_objective_mode: str = "point_tracking"
    target_assignment_mode: str = "legacy_nms"
    residual_regions_per_finger: int = 4
    residual_region_radius: float = 0.024
    candidate_pool_allow_multi_link_region: bool = False

    render: bool = True
    preview_finger: int = 1

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "CFETConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
