#!/usr/bin/env python3
"""Fail-closed compatibility and command-path audit for ACT static Shadow.

This file never starts a RealSense pipeline, never starts a Topic subscriber,
never imports/instantiates jogany_io, and never imports/instantiates GripperIO.
It only reads the explicitly approved Product v4 source files and performs
module-load compatibility checks without calling hardware-facing APIs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PROJECT_ROOT = Path("/home/robot/teleop_act/03_deployment")
PRODUCT_ROOT = Path(
    "/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v4_pro"
)
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
TOPIC_DIR = COLLECTOR_ROOT / "vendor" / "topic"
CHECKPOINT = PROJECT_ROOT / (
    "training_results/pick_place_object_v1/act/v1/train/20260827_113046/"
    "checkpoints/080000/pretrained_model"
)

APPROVED_PRODUCT_FILES = (
    COLLECTOR_ROOT / "config.py",
    COLLECTOR_ROOT / "topic_io.py",
    COLLECTOR_ROOT / "realsense.py",
    COLLECTOR_ROOT / "gripper_io.py",
    COLLECTOR_ROOT / "data_interface.py",
    PRODUCT_ROOT / "scripts" / "run_collect.sh",
)
TOPIC_READER_FILE = TOPIC_DIR / "tb6_topic_probe_reader.py"
TOPIC_LOADER_FILE = TOPIC_DIR / "platform_loader.py"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ShadowAssessment:
    python_executable: str
    python_version: str
    product_files_read: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    safe_for_static_shadow: bool = False

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_text(path: Path, assessment: ShadowAssessment) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing approved Product file: {path}")
    assessment.product_files_read.append(str(path))
    return path.read_text(encoding="utf-8")


def probe_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "imported")
    return True, str(version)


def probe_topic_extension() -> tuple[bool, str]:
    """Load only the native Python extension; do not call start_subscriber()."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LD_LIBRARY_PATH"] = str(TOPIC_DIR) + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
    )
    environment["PYTHONPATH"] = str(TOPIC_DIR) + (
        ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-c",
        "import topic; print(getattr(topic, '__file__', 'topic imported'))",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            env=environment,
        )
    except Exception as exc:
        return False, f"topic extension probe failed: {type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, result.stdout.strip() or "topic imported"
    return False, result.stderr.strip() or result.stdout.strip() or "topic import failed"


def checkpoint_files_ok() -> tuple[bool, str]:
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [filename for filename in required if not (CHECKPOINT / filename).is_file()]
    return (not missing, "all required checkpoint files present" if not missing else f"missing: {missing}")


def assess_environment() -> ShadowAssessment:
    assessment = ShadowAssessment(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
    )

    product_text: dict[Path, str] = {}
    for path in APPROVED_PRODUCT_FILES:
        product_text[path] = read_text(path, assessment)
    topic_reader = read_text(TOPIC_READER_FILE, assessment)
    topic_loader = read_text(TOPIC_LOADER_FILE, assessment)

    ok, detail = probe_import("lerobot")
    expected_lerobot = "0.6.1"
    if ok and detail == expected_lerobot:
        assessment.add("lerobot", "PASS", f"LeRobot {detail}")
    else:
        assessment.add("lerobot", "FAIL", detail)
        assessment.blockers.append(f"LeRobot {expected_lerobot} is not usable in this interpreter: {detail}")

    for module_name in ("cv2", "pyrealsense2"):
        ok, detail = probe_import(module_name)
        assessment.add(module_name, "PASS" if ok else "FAIL", detail)
        if not ok:
            assessment.blockers.append(
                f"Current Python cannot import {module_name}; Product RealSense capture cannot run here."
            )

    topic_ok, topic_detail = probe_topic_extension()
    assessment.add("product_topic_native_extension", "PASS" if topic_ok else "FAIL", topic_detail)
    if not topic_ok:
        assessment.blockers.append(
            "Current Python cannot load Product v4 topic.so; TB6 q_actual cannot be read in this process."
        )

    topic_safe = (
        "start_subscriber" in topic_reader
        and "get_system_state" in topic_reader
        and "jogany" not in topic_reader.lower()
        and "rpc" not in topic_reader.lower()
        and "jogany" not in topic_loader.lower()
        and "rpc" not in topic_loader.lower()
    )
    assessment.add(
        "topic_reader_command_path",
        "PASS" if topic_safe else "FAIL",
        "TB6ProbeTopicReader source exposes start_subscriber/get_system_state only; no Python RPC/JogAny path found."
        if topic_safe
        else "Topic reader source contains an unexpected RPC/JogAny reference.",
    )
    if not topic_safe:
        assessment.blockers.append("Topic reader command path is not provably read-only.")

    realsense_text = product_text[COLLECTOR_ROOT / "realsense.py"]
    camera_safe = (
        "rs.format.rgb8" in realsense_text
        and "cv2.COLOR_RGB2BGR" in realsense_text
        and "CAMERA_WIDTH = 640" in product_text[COLLECTOR_ROOT / "config.py"]
        and "CAMERA_HEIGHT = 480" in product_text[COLLECTOR_ROOT / "config.py"]
    )
    assessment.add(
        "training_input_image_contract",
        "PASS" if camera_safe else "FAIL",
        "Product capture config is 640x480 RGB8; JPEG conversion is RGB->BGR only for encoding. "
        "Raw get_frames() remains RGB arrays."
        if camera_safe
        else "Could not prove RGB8/640x480 raw camera contract from approved source.",
    )
    if not camera_safe:
        assessment.blockers.append("Cannot prove camera RGB/shape contract from Product capture source.")

    gripper_text = product_text[COLLECTOR_ROOT / "gripper_io.py"]
    unsafe_gripper = (
        "def start(" in gripper_text
        and "INITIAL_OPEN" in gripper_text
        and "driver.control(" in gripper_text
        and "def _best_effort_hold" in gripper_text
    )
    no_standalone_feedback_adapter = "class GripperFeedback" not in gripper_text and "def read_only" not in gripper_text
    assessment.add(
        "gripper_io_command_path",
        "FAIL" if unsafe_gripper else "PASS",
        "GripperIO.start() launches a worker that sends INITIAL_OPEN driver.control(...); shutdown may send hold control. "
        "It must not be instantiated by Shadow."
        if unsafe_gripper
        else "No GripperIO control call found in approved source.",
    )
    assessment.add(
        "gripper_read_only_feedback_path",
        "FAIL" if no_standalone_feedback_adapter else "PASS",
        "No standalone, existing read-only OmniPicker feedback adapter was found in the approved Product files/direct dependency. "
        "GripperIO.get_snapshot() has no live feedback unless its unsafe worker was started."
        if no_standalone_feedback_adapter
        else "A named standalone read-only gripper feedback adapter was found.",
    )
    if unsafe_gripper or no_standalone_feedback_adapter:
        assessment.blockers.append(
            "No safe existing gripper_actual feedback path: instantiating GripperIO would risk CAN control."
        )

    checkpoint_ok, checkpoint_detail = checkpoint_files_ok()
    assessment.add("checkpoint_080000", "PASS" if checkpoint_ok else "FAIL", checkpoint_detail)
    if not checkpoint_ok:
        assessment.blockers.append("Checkpoint 080000 is incomplete.")

    assessment.safe_for_static_shadow = not assessment.blockers
    return assessment


def format_report(assessment: ShadowAssessment) -> str:
    lines = [
        "TB6-R5 ACT v1 static Shadow environment audit",
        "=" * 45,
        f"Python executable: {assessment.python_executable}",
        f"Python version: {assessment.python_version}",
        "",
        "Product files read:",
        *[f"- {path}" for path in assessment.product_files_read],
        "",
        "Checks:",
        *[f"- [{check.status}] {check.name}: {check.detail}" for check in assessment.checks],
        "",
        "Blockers:",
        *([f"- {blocker}" for blocker in assessment.blockers] or ["- none"]),
        "",
        "Result: " + ("PASS: static Shadow may proceed." if assessment.safe_for_static_shadow else "NO-GO: hardware read/model inference is blocked before initialization."),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only static Shadow environment audit.")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    assessment = assess_environment()
    report = format_report(assessment)
    print(report, end="")
    if args.report_json is not None:
        output = args.report_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(assessment.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"[OUTPUT] {output}")
    return 0 if assessment.safe_for_static_shadow else 2


if __name__ == "__main__":
    raise SystemExit(main())
