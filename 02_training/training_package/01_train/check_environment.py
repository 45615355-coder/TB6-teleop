#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys

from packaging.version import Version

EXPECTED_LEROBOT = Version("0.6.1")
MIN_ACCELERATE = Version("1.14.0")
MAX_ACCELERATE = Version("2.0.0")


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print("=== TB6-R5 ACT training environment check ===")
    print(f"Python: {sys.version.split()[0]}")

    packages = ["lerobot", "torch", "torchvision", "datasets", "av", "accelerate", "wandb"]
    versions = {name: version(name) for name in packages}
    for name in packages:
        print(f"{name}: {versions[name]}")

    errors: list[str] = []

    try:
        if Version(versions["lerobot"]) != EXPECTED_LEROBOT:
            errors.append(f"LeRobot must be exactly {EXPECTED_LEROBOT}; got {versions['lerobot']}")
    except Exception:
        errors.append("LeRobot is not installed or version cannot be parsed.")

    try:
        acc = Version(versions["accelerate"])
        if not (MIN_ACCELERATE <= acc < MAX_ACCELERATE):
            errors.append(
                f"accelerate must satisfy >= {MIN_ACCELERATE} and < {MAX_ACCELERATE}; got {acc}"
            )
    except Exception:
        errors.append("accelerate is not installed or version cannot be parsed.")

    if versions["wandb"] == "NOT INSTALLED":
        errors.append("wandb is not installed (LeRobot training extra is incomplete).")

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            errors.append("CUDA is not available to PyTorch.")
        else:
            print(f"PyTorch CUDA runtime: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"GPU[{i}]: {props.name} | "
                    f"VRAM={props.total_memory / (1024**3):.2f} GiB"
                )
    except Exception as exc:
        errors.append(f"PyTorch import failed: {exc}")

    for exe in ["ffmpeg", "ffprobe", "lerobot-train"]:
        path = shutil.which(exe)
        print(f"{exe}: {path or 'NOT FOUND'}")
        if not path:
            errors.append(f"{exe} not found in PATH.")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        print("LeRobotDataset import: PASS")
    except Exception as exc:
        errors.append(f"LeRobotDataset import failed: {exc}")

    try:
        import accelerate  # noqa: F401
        import wandb  # noqa: F401
        print("Training-extra imports: PASS")
    except Exception as exc:
        errors.append(f"Training-extra import failed: {exc}")

    print()
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
