"""Convert TB6-R5 + OmniPicker episode PKLs to a LeRobot 0.6.1 dataset."""

import argparse
from pathlib import Path
import pickle

import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


CAMERA_NAMES = ("front", "top", "wrist")
AXIS_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]

FEATURES = {
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": AXIS_NAMES},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": AXIS_NAMES},
    },
}


def decode_jpeg(jpg_bytes):
    """Decode JPEG bytes from OpenCV BGR into an RGB numpy array."""
    encoded = np.frombuffer(jpg_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("JPEG image could not be decoded")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TB6-R5 + OmniPicker PKL episodes to LeRobot 0.6.1"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    pkl_files = sorted(args.data_dir.glob("*.pkl"))
    if args.num_episodes is not None:
        pkl_files = pkl_files[: args.num_episodes]
    if not pkl_files:
        raise ValueError(f"No PKL files found in {args.data_dir}")

    print(f"Found {len(pkl_files)} PKL episodes")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=50,
        features=FEATURES,
        root=args.output_root,
        robot_type="tb6_r5_omnipicker",
        use_videos=True,
    )

    for episode_index, pkl_file in enumerate(pkl_files, start=1):
        print(f"[{episode_index}/{len(pkl_files)}] Processing: {pkl_file.name}")
        with pkl_file.open("rb") as file:
            episode_data = pickle.load(file)

        for step_data in episode_data:
            state = np.concatenate(
                [
                    np.asarray(step_data["q_actual"], dtype=np.float32),
                    np.array([step_data["gripper_actual_pos"]], dtype=np.float32),
                ]
            )
            action = np.concatenate(
                [
                    np.asarray(step_data["q_command"], dtype=np.float32),
                    np.array([step_data["gripper_command_pos"]], dtype=np.float32),
                ]
            )

            images = {}
            for camera_name in CAMERA_NAMES:
                try:
                    images[camera_name] = decode_jpeg(
                        step_data["image"][camera_name]["color"]
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Could not decode {camera_name} image in {pkl_file.name}"
                    ) from error

            frame = {
                "observation.images.front": images["front"],
                "observation.images.top": images["top"],
                "observation.images.wrist": images["wrist"],
                "observation.state": state,
                "action": action,
                "task": args.task,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"Saved episode: {pkl_file.name} ({len(episode_data)} steps)")

    dataset.finalize()
    print("Conversion complete")
    print(f"episodes: {len(pkl_files)}")
    print(f"output: {args.output_root}")


if __name__ == "__main__":
    main()
