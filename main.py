from __future__ import annotations

import argparse
import json

from method.config import CFETConfig
from method.sequential_qp import SequentialQP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F2M local method runner")
    parser.add_argument("command", choices=["responsibility", "sequential_qp"])
    parser.add_argument("--config", default="configs/cfet_default.json")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--robot-name")
    parser.add_argument("--urdf", dest="urdf_path")
    parser.add_argument("--robot-link-pc", dest="robot_link_pc_path")
    parser.add_argument("--point-cloud", dest="object_point_cloud")
    parser.add_argument("--initial-q", dest="initial_q_path")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--mode")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--qp-iters", type=int)
    parser.add_argument("--palm-qp-iters", type=int)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction)
    parser.add_argument("--preview-finger", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CFETConfig.from_json(
        args.config,
        object_name=args.object_name,
        robot_name=args.robot_name,
        urdf_path=args.urdf_path,
        robot_link_pc_path=args.robot_link_pc_path,
        object_point_cloud=args.object_point_cloud,
        initial_q_path=args.initial_q_path,
        modes=args.modes or ([args.mode] if args.mode else None),
        output_dir=args.output_dir,
        device=args.device,
        cycles=args.cycles,
        qp_iters=args.qp_iters,
        palm_qp_iters=args.palm_qp_iters,
        render=args.render,
        preview_finger=args.preview_finger,
    )
    runner = SequentialQP(config)

    if args.command == "responsibility":
        mode = config.modes[0]
        report = runner.inspect_responsibility(mode, config.preview_finger)
        print(json.dumps(report, indent=2))
        return

    results = runner.run_all_modes()
    print(
        json.dumps(
            {
                "object": config.object_name,
                "modes": config.modes,
                "output_dir": config.output_dir,
                "num_results": len(results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
