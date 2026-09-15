#!/usr/bin/env python3
"""Read-only, offline evaluation of local LeRobot ACT checkpoints.

This module deliberately contains no robot, teleoperator, or environment imports.
It only decodes observations from a local LeRobotDataset and runs local policy
inference. Dataset statistics are never supplied to the policy processors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Enforce local-only behavior before importing Hugging Face / LeRobot modules.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(tempfile.gettempdir()) / "tb6_act_offline_evaluation_hf_cache"),
)

import numpy as np
import torch

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


EXPECTED_LEROBOT_VERSION = "0.6.1"
EXPECTED_EPISODES = 10
EXPECTED_FRAMES = 20641
EXPECTED_FPS = 50
EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
EXPECTED_CHUNK_SIZE = 100
EXPECTED_N_ACTION_STEPS = 100
EXPECTED_N_OBS_STEPS = 1
EXPECTED_CAMERAS = (
    "observation.images.front",
    "observation.images.top",
    "observation.images.wrist",
)
DEFAULT_CHECKPOINTS = ("020000", "040000", "060000", "080000", "100000","120000")
DEFAULT_DATASET_ROOT = Path(
    "/home/robot/teleop_act/01_teleoperation/quest3_tb6_r5_omnipicker_data_product_v6/"
    "data_teleop/pick_place_object_v2/test_datasets/pick_place_object_dataset"
)
DEFAULT_CHECKPOINTS_ROOT = Path(
    "/home/robot/data_collect/training_results/pick_place_object_v2/act/v1/"
    "train/20260901_172324/checkpoints"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/robot/teleop_act/03_deployment/logs/evaluation"
)


class EvaluationError(RuntimeError):
    """A fatal validation or inference error."""


@dataclass
class ErrorAccumulator:
    checkpoint: str
    episode_index: int | None = None
    valid_action_count: int = 0
    _parts: list[np.ndarray] = field(default_factory=list)

    def add(self, absolute_error: np.ndarray) -> None:
        if absolute_error.ndim != 2 or absolute_error.shape[1] != EXPECTED_ACTION_DIM:
            raise EvaluationError(
                f"Internal error accumulator received shape {absolute_error.shape}, expected [N, 7]"
            )
        if not np.isfinite(absolute_error).all():
            raise EvaluationError("Absolute-error array contains NaN/Inf")
        self._parts.append(absolute_error)
        self.valid_action_count += int(absolute_error.shape[0])

    def errors(self) -> np.ndarray:
        if not self._parts:
            raise EvaluationError(
                f"No valid actions accumulated for checkpoint={self.checkpoint}, "
                f"episode={self.episode_index}"
            )
        return np.concatenate(self._parts, axis=0)

    def per_episode_row(self) -> dict[str, Any]:
        error = self.errors()
        arm_deg = np.rad2deg(error[:, :6])
        return {
            "checkpoint": self.checkpoint,
            "episode_index": self.episode_index,
            "valid_action_count": self.valid_action_count,
            "arm_mae_deg": float(arm_deg.mean()),
            "gripper_mae_pct": float(error[:, 6].mean()),
        }

    def summary_row(self, num_episodes: int) -> dict[str, Any]:
        error = self.errors()
        arm_deg = np.rad2deg(error[:, :6])
        row: dict[str, Any] = {
            "checkpoint": self.checkpoint,
            "num_episodes": num_episodes,
            "valid_action_count": self.valid_action_count,
        }
        for joint_index in range(6):
            row[f"joint_{joint_index + 1}_mae_deg"] = float(arm_deg[:, joint_index].mean())
        row.update(
            {
                "arm_mae_deg": float(arm_deg.mean()),
                "arm_p95_deg": float(np.percentile(arm_deg, 95)),
                "arm_max_deg": float(arm_deg.max()),
                "gripper_mae_pct": float(error[:, 6].mean()),
                "gripper_p95_pct": float(np.percentile(error[:, 6], 95)),
                "gripper_max_pct": float(error[:, 6].max()),
            }
        )
        return row


SUMMARY_FIELDS = (
    "checkpoint",
    "num_episodes",
    "valid_action_count",
    "joint_1_mae_deg",
    "joint_2_mae_deg",
    "joint_3_mae_deg",
    "joint_4_mae_deg",
    "joint_5_mae_deg",
    "joint_6_mae_deg",
    "arm_mae_deg",
    "arm_p95_deg",
    "arm_max_deg",
    "gripper_mae_pct",
    "gripper_p95_pct",
    "gripper_max_pct",
)
PER_EPISODE_FIELDS = (
    "checkpoint",
    "episode_index",
    "valid_action_count",
    "arm_mae_deg",
    "gripper_mae_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline ACT v1 checkpoint evaluation (no robot connection or control)."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="Checkpoint directory names under --checkpoints-root.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="Episode indices to evaluate. Omit to evaluate all 10 episodes.",
    )
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except Exception as exc:
        raise EvaluationError(f"Failed to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected a JSON object in {path}")
    return value


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise EvaluationError(f"Missing or empty required file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise EvaluationError(f"Missing required directory: {path}")


def validate_processor_files(pretrained_model: Path, filename: str, required_step: str) -> None:
    config_path = pretrained_model / filename
    require_file(config_path)
    config = load_json(config_path)
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EvaluationError(f"Processor has no steps: {config_path}")
    registry_names = [step.get("registry_name") for step in steps if isinstance(step, dict)]
    if required_step not in registry_names:
        raise EvaluationError(f"{config_path} does not contain required step {required_step!r}")
    for step in steps:
        if not isinstance(step, dict):
            raise EvaluationError(f"Invalid processor step in {config_path}: {step!r}")
        state_file = step.get("state_file")
        if state_file is not None:
            if not isinstance(state_file, str) or Path(state_file).name != state_file:
                raise EvaluationError(f"Unsafe/invalid processor state_file in {config_path}: {state_file!r}")
            require_file(pretrained_model / state_file)


def config_action_dim(config: dict[str, Any]) -> int | None:
    shape = config.get("output_features", {}).get("action", {}).get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def config_cameras(config: dict[str, Any]) -> tuple[str, ...]:
    features = config.get("input_features", {})
    return tuple(key for key in features if key.startswith("observation.images."))


def validate_checkpoint(pretrained_model: Path) -> dict[str, Any]:
    require_dir(pretrained_model)
    for filename in ("config.json", "model.safetensors", "train_config.json"):
        require_file(pretrained_model / filename)
    validate_processor_files(
        pretrained_model, "policy_preprocessor.json", "normalizer_processor"
    )
    validate_processor_files(
        pretrained_model, "policy_postprocessor.json", "unnormalizer_processor"
    )

    config = load_json(pretrained_model / "config.json")
    expected_values = {
        "type": "act",
        "chunk_size": EXPECTED_CHUNK_SIZE,
        "n_action_steps": EXPECTED_N_ACTION_STEPS,
        "n_obs_steps": EXPECTED_N_OBS_STEPS,
        "temporal_ensemble_coeff": None,
    }
    for key, expected in expected_values.items():
        actual = config.get(key)
        if actual != expected:
            raise EvaluationError(
                f"Invalid ACT config in {pretrained_model / 'config.json'}: "
                f"{key}={actual!r}, expected {expected!r}"
            )
    if config_action_dim(config) != EXPECTED_ACTION_DIM:
        raise EvaluationError(
            f"Checkpoint action dimension must be 7: {pretrained_model / 'config.json'}"
        )
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    if state_shape != [EXPECTED_STATE_DIM]:
        raise EvaluationError(
            f"Checkpoint observation.state dimension must be 7: {pretrained_model / 'config.json'}"
        )
    cameras = config_cameras(config)
    if set(cameras) != set(EXPECTED_CAMERAS) or len(cameras) != len(EXPECTED_CAMERAS):
        raise EvaluationError(
            f"Checkpoint cameras must be exactly {EXPECTED_CAMERAS}, got {cameras}: "
            f"{pretrained_model / 'config.json'}"
        )
    return config


def validate_dataset_metadata(dataset_root: Path) -> dict[str, Any]:
    require_dir(dataset_root)
    required_paths = (
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "stats.json",
        dataset_root / "meta" / "tasks.parquet",
        dataset_root / "meta" / "episodes",
        dataset_root / "data",
        dataset_root / "videos",
    )
    for path in required_paths:
        if path.suffix:
            require_file(path)
        else:
            require_dir(path)

    info = load_json(dataset_root / "meta" / "info.json")
    checks = {
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "fps": EXPECTED_FPS,
    }
    for key, expected in checks.items():
        if info.get(key) != expected:
            raise EvaluationError(
                f"Dataset {key}={info.get(key)!r}, expected {expected}: {dataset_root}"
            )

    features = info.get("features", {})
    for camera in EXPECTED_CAMERAS:
        spec = features.get(camera)
        if not isinstance(spec, dict) or spec.get("dtype") != "video":
            raise EvaluationError(f"Dataset is missing required video camera feature: {camera}")
        require_dir(dataset_root / "videos" / camera)
    if features.get("observation.state", {}).get("shape") != [EXPECTED_STATE_DIM]:
        raise EvaluationError("Dataset observation.state must be 7D")
    if features.get("action", {}).get("shape") != [EXPECTED_ACTION_DIM]:
        raise EvaluationError("Dataset action must be 7D")
    return info


def normalize_requested_values(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    checkpoints = list(dict.fromkeys(args.checkpoints))
    if not checkpoints:
        raise EvaluationError("At least one checkpoint is required")
    for checkpoint in checkpoints:
        if not checkpoint.isdigit() or Path(checkpoint).name != checkpoint:
            raise EvaluationError(f"Invalid checkpoint name: {checkpoint!r}")

    episodes = list(range(EXPECTED_EPISODES)) if args.episodes is None else list(dict.fromkeys(args.episodes))
    if not episodes:
        raise EvaluationError("At least one episode is required")
    invalid = [episode for episode in episodes if episode < 0 or episode >= EXPECTED_EPISODES]
    if invalid:
        raise EvaluationError(f"Episode indices out of range [0, 9]: {invalid}")
    return checkpoints, episodes


def make_dataset(dataset_root: Path, video_backend: str) -> LeRobotDataset:
    delta_timestamps = {
        "action": [step / EXPECTED_FPS for step in range(EXPECTED_CHUNK_SIZE)]
    }
    try:
        # The local files were preflighted above. download_videos=False and offline
        # environment variables prevent network fallback.
        dataset = LeRobotDataset(
            repo_id="local/pick_place_object_v2_test",
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            download_videos=False,
        )
    except Exception as exc:
        raise EvaluationError(f"Failed to load local validation Dataset: {exc}") from exc

    if dataset.meta.total_episodes != EXPECTED_EPISODES or dataset.num_episodes != EXPECTED_EPISODES:
        raise EvaluationError(
            f"Loaded Dataset episode count mismatch: metadata={dataset.meta.total_episodes}, "
            f"selected={dataset.num_episodes}, expected={EXPECTED_EPISODES}"
        )
    if dataset.meta.total_frames != EXPECTED_FRAMES or len(dataset) != EXPECTED_FRAMES:
        raise EvaluationError(
            f"Loaded Dataset frame count mismatch: metadata={dataset.meta.total_frames}, "
            f"loaded={len(dataset)}, expected={EXPECTED_FRAMES}"
        )
    if dataset.meta.fps != EXPECTED_FPS:
        raise EvaluationError(f"Loaded Dataset fps={dataset.meta.fps}, expected={EXPECTED_FPS}")
    return dataset


def load_policy_and_processors(
    pretrained_model: Path, device: torch.device
) -> tuple[ACTPolicy, Any, Any, ACTConfig]:
    try:
        config = ACTConfig.from_pretrained(
            pretrained_name_or_path=str(pretrained_model),
            local_files_only=True,
        )
        config.device = str(device)
        # This affects only constructor initialization. strict checkpoint loading
        # immediately replaces every model tensor and avoids a network/cache lookup.
        config.pretrained_backbone_weights = None
        policy = ACTPolicy.from_pretrained(
            pretrained_name_or_path=str(pretrained_model),
            config=config,
            local_files_only=True,
            strict=True,
        )
        policy.eval()
    except Exception as exc:
        raise EvaluationError(f"Checkpoint model loading failed for {pretrained_model}: {exc}") from exc

    try:
        # Do not pass dataset_stats or dataset_meta: all normalization state must
        # come from this checkpoint's serialized processor files.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(pretrained_model),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except Exception as exc:
        raise EvaluationError(
            f"Checkpoint pre/postprocessor loading failed for {pretrained_model}: {exc}"
        ) from exc
    return policy, preprocessor, postprocessor, config


def scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    try:
        episode = dataset.meta.episodes[episode_index]
        actual_index = scalar_int(episode["episode_index"])
        start = scalar_int(episode["dataset_from_index"])
        end = scalar_int(episode["dataset_to_index"])
    except Exception as exc:
        raise EvaluationError(f"Failed to read metadata for episode {episode_index}: {exc}") from exc
    if actual_index != episode_index:
        raise EvaluationError(
            f"Episode metadata ordering mismatch: requested={episode_index}, row={actual_index}"
        )
    if start < 0 or end <= start or end > EXPECTED_FRAMES:
        raise EvaluationError(f"Invalid episode {episode_index} bounds: [{start}, {end})")
    return start, end


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise EvaluationError(f"{name} is not a torch.Tensor: {type(tensor)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise EvaluationError(f"{name} contains NaN/Inf")


def evaluate_episode(
    checkpoint: str,
    episode_index: int,
    dataset: LeRobotDataset,
    policy: ACTPolicy,
    preprocessor: Any,
    postprocessor: Any,
) -> ErrorAccumulator:
    start, end = episode_bounds(dataset, episode_index)
    accumulator = ErrorAccumulator(checkpoint=checkpoint, episode_index=episode_index)

    for absolute_start in range(start, end, EXPECTED_CHUNK_SIZE):
        try:
            item = dataset[absolute_start]
        except Exception as exc:
            raise EvaluationError(
                f"Dataset decoding failed at episode={episode_index}, index={absolute_start}: {exc}"
            ) from exc

        if scalar_int(item.get("episode_index")) != episode_index:
            raise EvaluationError(
                f"Dataset index {absolute_start} belongs to the wrong episode: "
                f"{scalar_int(item.get('episode_index'))} != {episode_index}"
            )
        expected_frame_index = absolute_start - start
        if scalar_int(item.get("frame_index")) != expected_frame_index:
            raise EvaluationError(
                f"Unexpected frame_index at absolute index {absolute_start}: "
                f"{scalar_int(item.get('frame_index'))} != {expected_frame_index}"
            )

        observation: dict[str, torch.Tensor] = {}
        for key in ("observation.state", *EXPECTED_CAMERAS):
            if key not in item:
                raise EvaluationError(f"Dataset item is missing required feature {key}")
            require_finite_tensor(f"Dataset {key}", item[key])
            observation[key] = item[key]
        if tuple(observation["observation.state"].shape) != (EXPECTED_STATE_DIM,):
            raise EvaluationError(
                f"observation.state shape={tuple(observation['observation.state'].shape)}, expected=(7,)"
            )

        target = item.get("action")
        padding = item.get("action_is_pad")
        require_finite_tensor("Dataset action chunk", target)
        if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool:
            raise EvaluationError("Dataset action_is_pad is missing or is not a bool tensor")
        if tuple(target.shape) != (EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
            raise EvaluationError(
                f"Ground-truth action shape={tuple(target.shape)}, expected=(100, 7)"
            )
        if tuple(padding.shape) != (EXPECTED_CHUNK_SIZE,):
            raise EvaluationError(f"action_is_pad shape={tuple(padding.shape)}, expected=(100,)")

        try:
            processed_observation = preprocessor(observation)
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(processed_observation)
            require_finite_tensor("Normalized predicted action chunk", prediction)
            prediction = postprocessor(prediction)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(
                f"Inference failed for checkpoint={checkpoint}, episode={episode_index}, "
                f"frame={expected_frame_index}: {exc}"
            ) from exc

        require_finite_tensor("Postprocessed predicted action chunk", prediction)
        if tuple(prediction.shape) != (1, EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
            raise EvaluationError(
                f"Predicted action shape={tuple(prediction.shape)}, expected=(1, 100, 7)"
            )
        prediction = prediction.squeeze(0).detach().cpu()
        target = target.detach().cpu()
        if tuple(prediction.shape) != tuple(target.shape):
            raise EvaluationError(
                f"Prediction/target action shape mismatch: {tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        valid_mask = ~padding.detach().cpu()
        expected_valid = min(EXPECTED_CHUNK_SIZE, end - absolute_start)
        valid_count = int(valid_mask.sum().item())
        if valid_count != expected_valid:
            raise EvaluationError(
                f"Padding mask valid count mismatch at episode={episode_index}, "
                f"frame={expected_frame_index}: {valid_count} != {expected_valid}"
            )
        absolute_error = torch.abs(prediction[valid_mask] - target[valid_mask]).numpy()
        accumulator.add(absolute_error)

    expected_episode_actions = end - start
    if accumulator.valid_action_count != expected_episode_actions:
        raise EvaluationError(
            f"Episode {episode_index} valid action count={accumulator.valid_action_count}, "
            f"expected={expected_episode_actions}"
        )
    return accumulator


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: (f"{value:.10f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(formatted)


def make_report(
    output_dir: Path,
    dataset_root: Path,
    checkpoints_root: Path,
    episodes: list[int],
    device: torch.device,
    summary_rows: list[dict[str, Any]],
    per_episode_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "TB6-R5 ACT v1 offline checkpoint evaluation",
        "=" * 47,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"LeRobot: {lerobot.__version__}",
        f"torch: {torch.__version__}",
        f"Device: {device}",
        f"Dataset: {dataset_root}",
        f"Dataset: {EXPECTED_EPISODES} episodes, {EXPECTED_FRAMES} frames, {EXPECTED_FPS} fps",
        f"Checkpoints root: {checkpoints_root}",
        f"Evaluated episodes: {episodes}",
        "",
        "Objective metrics (no composite score)",
        "-" * 38,
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"Checkpoint {row['checkpoint']}",
                f"  episodes / valid actions : {row['num_episodes']} / {row['valid_action_count']}",
                "  joint MAE (degree)        : "
                + ", ".join(
                    f"J{i}={row[f'joint_{i}_mae_deg']:.6f}" for i in range(1, 7)
                ),
                f"  arm pooled MAE / P95 / max (degree): "
                f"{row['arm_mae_deg']:.6f} / {row['arm_p95_deg']:.6f} / {row['arm_max_deg']:.6f}",
                f"  gripper MAE / P95 / max (percentage point): "
                f"{row['gripper_mae_pct']:.6f} / {row['gripper_p95_pct']:.6f} / "
                f"{row['gripper_max_pct']:.6f}",
                "",
            ]
        )
    lines.extend(["Per-episode metrics", "-" * 19])
    for row in per_episode_rows:
        lines.append(
            f"checkpoint={row['checkpoint']} episode={row['episode_index']} "
            f"valid_actions={row['valid_action_count']} arm_mae_deg={row['arm_mae_deg']:.6f} "
            f"gripper_mae_pct={row['gripper_mae_pct']:.6f}"
        )
    lines.extend(
        [
            "",
            "Evaluation semantics",
            "-" * 20,
            "Each episode is partitioned at frame 0, 100, 200, ... . At each chunk start,",
            "only the current 7D state and the current front/top/wrist images are passed through",
            "the checkpoint's saved preprocessor. Official ACT predict_action_chunk() predicts",
            "100 normalized actions, and that checkpoint's saved postprocessor restores original",
            "units. The Dataset action_is_pad mask excludes tail padding from every metric.",
            "Validation Dataset stats.json is not used for model normalization.",
        ]
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_runtime_header(
    dataset_root: Path,
    info: dict[str, Any],
    checkpoints: list[str],
    checkpoint_models: dict[str, Path],
    checkpoint_configs: dict[str, dict[str, Any]],
    episodes: list[int],
    device: torch.device,
) -> None:
    print("=== TB6-R5 ACT v1 offline checkpoint evaluation ===")
    print(f"LeRobot version : {lerobot.__version__}")
    print(f"torch version   : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU             : N/A (using CPU)")
    print(f"Device          : {device}")
    print(f"Dataset root    : {dataset_root}")
    print(
        f"Dataset         : episodes={info['total_episodes']} "
        f"frames={info['total_frames']} fps={info['fps']}"
    )
    print(f"Checkpoints     : {', '.join(checkpoints)}")
    print(f"Episodes        : {episodes}")
    for checkpoint in checkpoints:
        config = checkpoint_configs[checkpoint]
        print(f"[{checkpoint}] pretrained_model: {checkpoint_models[checkpoint]}")
        print(
            f"[{checkpoint}] model: chunk_size={config['chunk_size']} "
            f"n_action_steps={config['n_action_steps']} "
            f"action_dim={config_action_dim(config)} "
            f"cameras={list(config_cameras(config))}"
        )
    print("Safety          : local Dataset + local checkpoints; no robot/control interfaces")
    print()


def run() -> Path:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoints, episodes = normalize_requested_values(args)

    if lerobot.__version__ != EXPECTED_LEROBOT_VERSION:
        raise EvaluationError(
            f"LeRobot version must be {EXPECTED_LEROBOT_VERSION}, got {lerobot.__version__}"
        )
    info = validate_dataset_metadata(dataset_root)
    require_dir(checkpoints_root)

    checkpoint_models: dict[str, Path] = {}
    checkpoint_configs: dict[str, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        pretrained_model = checkpoints_root / checkpoint / "pretrained_model"
        checkpoint_models[checkpoint] = pretrained_model
        checkpoint_configs[checkpoint] = validate_checkpoint(pretrained_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_runtime_header(
        dataset_root,
        info,
        checkpoints,
        checkpoint_models,
        checkpoint_configs,
        episodes,
        device,
    )
    dataset = make_dataset(dataset_root, args.video_backend)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / stamp
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        raise EvaluationError(f"Failed to create output directory {output_dir}: {exc}") from exc
    print(f"Output directory: {output_dir}")

    summary_rows: list[dict[str, Any]] = []
    per_episode_rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        pretrained_model = checkpoint_models[checkpoint]
        print(f"\n[{checkpoint}] Loading local ACT policy and checkpoint processors ...", flush=True)
        policy, preprocessor, postprocessor, runtime_config = load_policy_and_processors(
            pretrained_model, device
        )
        if (
            runtime_config.chunk_size != EXPECTED_CHUNK_SIZE
            or runtime_config.n_action_steps != EXPECTED_N_ACTION_STEPS
            or runtime_config.n_obs_steps != EXPECTED_N_OBS_STEPS
            or runtime_config.temporal_ensemble_coeff is not None
        ):
            raise EvaluationError(f"Runtime ACT config changed unexpectedly for {checkpoint}")

        overall = ErrorAccumulator(checkpoint=checkpoint)
        for episode_index in episodes:
            print(f"[{checkpoint}] Evaluating episode {episode_index} ...", flush=True)
            episode_accumulator = evaluate_episode(
                checkpoint,
                episode_index,
                dataset,
                policy,
                preprocessor,
                postprocessor,
            )
            overall.add(episode_accumulator.errors())
            row = episode_accumulator.per_episode_row()
            per_episode_rows.append(row)
            print(
                f"[{checkpoint}] episode={episode_index} "
                f"valid_actions={row['valid_action_count']} "
                f"arm_mae_deg={row['arm_mae_deg']:.6f} "
                f"gripper_mae_pct={row['gripper_mae_pct']:.6f}",
                flush=True,
            )

        summary = overall.summary_row(num_episodes=len(episodes))
        summary_rows.append(summary)
        print(
            f"[{checkpoint}] complete: valid_actions={summary['valid_action_count']} "
            f"arm_mae_deg={summary['arm_mae_deg']:.6f} "
            f"gripper_mae_pct={summary['gripper_mae_pct']:.6f}",
            flush=True,
        )

        del policy, preprocessor, postprocessor, runtime_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(output_dir / "per_episode.csv", PER_EPISODE_FIELDS, per_episode_rows)
    make_report(
        output_dir,
        dataset_root,
        checkpoints_root,
        episodes,
        device,
        summary_rows,
        per_episode_rows,
    )
    print(f"\n[PASS] Offline evaluation completed: {output_dir}")
    print(f"[OUTPUT] {output_dir / 'summary.csv'}")
    print(f"[OUTPUT] {output_dir / 'per_episode.csv'}")
    print(f"[OUTPUT] {output_dir / 'report.txt'}")
    return output_dir


def main() -> int:
    try:
        run()
    except (EvaluationError, FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[FAIL] Evaluation interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
