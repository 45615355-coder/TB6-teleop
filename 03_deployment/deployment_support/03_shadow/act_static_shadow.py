#!/usr/bin/env python3
"""Fail-closed entry point for TB6-R5 ACT v1 static Shadow.

Current Product v4 lacks a provably safe gripper_actual feedback path and its
hardware Python dependencies are incompatible with LeRobot Python 3.12. This
entry point records that NO-GO state and exits before any camera, Topic, CAN,
robot, policy, or processor initialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

from check_shadow_environment import assess_environment, format_report


DEFAULT_OUTPUT_ROOT = Path(
"/home/robot/teleop_act/03_deployment/logs/shadow/static"
)


class ShadowSafetyError(RuntimeError):
    """Raised before any hardware or model initialization when Shadow is not safe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB6-R5 ACT v1 static Shadow safety gate.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assessment = assess_environment()
    output_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "shadow_environment_report.txt").write_text(
        format_report(assessment), encoding="utf-8"
    )
    (output_dir / "shadow_environment_report.json").write_text(
        json.dumps(assessment.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print("[SHADOW]")
    print("READ ONLY")
    print("NO RPC COMMAND")
    print("NO JOGANY")
    print("NO GRIPPER COMMAND")
    print("ROBOT WILL NOT BE MOVED BY ACT")
    print(f"[OUTPUT] {output_dir}")
    print(format_report(assessment), end="")

    if not assessment.safe_for_static_shadow:
        print("[NO-GO] No hardware object, no ACT policy, and no processor was initialized.")
        print("[NO-GO] prediction.npz/csv were intentionally not generated: observation.state requires safe gripper_actual.")
        return 2

    # Do not add an implicit fallback here. A future implementation must first
    # identify and review an existing Product read-only gripper feedback adapter.
    raise ShadowSafetyError(
        "No runtime backend is enabled. Static Shadow requires an explicitly reviewed read-only gripper feedback adapter."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShadowSafetyError as exc:
        print(f"[NO-GO] {exc}", file=sys.stderr)
        raise SystemExit(2)
