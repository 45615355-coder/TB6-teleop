#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_CAMERAS = (
    "observation.images.front",
    "observation.images.top",
    "observation.images.wrist",
)
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 107014
EXPECTED_FPS = 50
EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
EXPECTED_ROBOT_TYPE = "tb6_r5_omnipicker"


def shape_tuple(x):
    return tuple(int(v) for v in x.shape)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only validation of the TB6-R5 LeRobot ACT training dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_dir = root / "meta" / "episodes"
    data_dir = root / "data"
    videos_dir = root / "videos"

    print("=== TB6-R5 LeRobot dataset check ===")
    print(f"Dataset root: {root}")

    errors = []
    required = [info_path, stats_path, tasks_path, episodes_dir, data_dir, videos_dir]
    for p in required:
        if not p.exists():
            errors.append(f"Missing required path: {p}")

    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})

    checks = [
        ("codebase_version", info.get("codebase_version"), "v3.0"),
        ("fps", info.get("fps"), EXPECTED_FPS),
        ("total_episodes", info.get("total_episodes"), EXPECTED_EPISODES),
        ("total_frames", info.get("total_frames"), EXPECTED_FRAMES),
        ("total_tasks", info.get("total_tasks"), 1),
        ("robot_type", info.get("robot_type"), EXPECTED_ROBOT_TYPE),
    ]
    for name, actual, expected in checks:
        ok = actual == expected
        print(f"{name}: {actual!r} | expected={expected!r} | {'PASS' if ok else 'FAIL'}")
        if not ok:
            errors.append(f"{name}: expected {expected!r}, got {actual!r}")

    for cam in EXPECTED_CAMERAS:
        spec = features.get(cam)
        if spec is None:
            errors.append(f"Missing feature: {cam}")
            continue
        ok = spec.get("dtype") == "video" and spec.get("shape") == [480, 640, 3]
        print(
            f"{cam}: dtype={spec.get('dtype')} shape={spec.get('shape')} | "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            errors.append(f"Unexpected camera feature spec for {cam}: {spec}")

    state_spec = features.get("observation.state")
    action_spec = features.get("action")
    if state_spec is None or state_spec.get("shape") != [EXPECTED_STATE_DIM]:
        errors.append(f"observation.state must have shape [{EXPECTED_STATE_DIM}]")
    if action_spec is None or action_spec.get("shape") != [EXPECTED_ACTION_DIM]:
        errors.append(f"action must have shape [{EXPECTED_ACTION_DIM}]")

    print(f"observation.state spec: {state_spec}")
    print(f"action spec: {action_spec}")

    if errors:
        print()
        print("Metadata RESULT: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    print()
    print("Loading dataset through LeRobot 0.6.1 ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="local/tb6_r5_pick_place_object_v2_train",
        root=root,
        video_backend="pyav",
    )

    print(dataset)
    if len(dataset) != EXPECTED_FRAMES:
        errors.append(f"LeRobotDataset len={len(dataset)}, expected={EXPECTED_FRAMES}")
    if dataset.num_episodes != EXPECTED_EPISODES:
        errors.append(
            f"LeRobotDataset num_episodes={dataset.num_episodes}, expected={EXPECTED_EPISODES}"
        )

    # Read raw parquet-backed columns without video decoding.
    raw = dataset.hf_dataset
    if raw.num_rows != EXPECTED_FRAMES:
        errors.append(f"HF dataset rows={raw.num_rows}, expected={EXPECTED_FRAMES}")

    # Decode a small deterministic set of frames, including all three cameras.
    sample_indices = sorted(set([
        0,
        EXPECTED_FRAMES // 4,
        EXPECTED_FRAMES // 2,
        (3 * EXPECTED_FRAMES) // 4,
        EXPECTED_FRAMES - 1,
    ]))

    print()
    print(f"Decoding sample frames: {sample_indices}")
    for idx in sample_indices:
        item = dataset[idx]

        state = item["observation.state"]
        action = item["action"]

        if shape_tuple(state) != (EXPECTED_STATE_DIM,):
            errors.append(f"frame {idx}: state shape={shape_tuple(state)}")
        if shape_tuple(action) != (EXPECTED_ACTION_DIM,):
            errors.append(f"frame {idx}: action shape={shape_tuple(action)}")

        if not bool(state.isfinite().all()):
            errors.append(f"frame {idx}: observation.state contains NaN/Inf")
        if not bool(action.isfinite().all()):
            errors.append(f"frame {idx}: action contains NaN/Inf")

        camera_shapes = {}
        for cam in EXPECTED_CAMERAS:
            image = item[cam]
            camera_shapes[cam] = shape_tuple(image)
            if shape_tuple(image) != (3, 480, 640):
                errors.append(
                    f"frame {idx}: {cam} decoded shape={shape_tuple(image)}, expected=(3,480,640)"
                )
            if not bool(image.isfinite().all()):
                errors.append(f"frame {idx}: {cam} contains NaN/Inf")

        episode_index = item.get("episode_index")
        frame_index = item.get("frame_index")
        print(
            f"frame={idx} episode={episode_index} frame_index={frame_index} "
            f"state={shape_tuple(state)} action={shape_tuple(action)} "
            f"images={camera_shapes}"
        )

    print()
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    print("RESULT: PASS")
    print(
        "Confirmed: 50 episodes, 107014 frames, 50 Hz, "
        "3 RGB cameras, 7D state, 7D action, and successful video decoding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
